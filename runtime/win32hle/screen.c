/*
 * screen.c - the display and input for a DirectDraw game on win32hle: an SDL2
 * window showing the primary surface (ddraw.c), scaled to the window with its
 * aspect kept, and SDL's mouse and keyboard turned into the window messages
 * a game's windows get (user32.c's hle_input_*).
 *
 * It runs as user32's pump hook, on the thread that runs the game (SDL wants
 * its video calls on one thread, and the game's thread is the one that pumps
 * messages). A frame is shown when the primary has changed, at most 60 times
 * a second: from the pump, and as the game draws into the primary (ddraw.c's
 * hle_dd_primary_hook), since on Windows the primary is the screen. A game
 * that draws without pumping (a loading screen, a menu's transition) is seen
 * as it draws, as on Windows.
 *
 * Headless (hle_screen_open(..., headless)), there is no window: the pump
 * only paces, and input comes from a host's script.
 */
#define _GNU_SOURCE
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_vulkan.h>
#include "win32hle.h"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

#define WM_MOUSEMOVE   0x0200u
#define WM_LBUTTONDOWN 0x0201u
#define WM_LBUTTONUP   0x0202u
#define WM_RBUTTONDOWN 0x0204u
#define WM_RBUTTONUP   0x0205u
#define WM_MBUTTONDOWN 0x0207u
#define WM_MBUTTONUP   0x0208u
#define WM_CLOSE       0x0010u

static SDL_Window *g_win;
static SDL_Renderer *g_ren;
static SDL_Texture *g_tex;
static int g_tex_w, g_tex_h, g_tex_fmt;
static unsigned g_shown_seq = ~0u;
static uint32_t g_last_present;
static int g_headless, g_fullscreen;
static SDL_Rect g_dst;                                   /* where the picture is in the window */
/* Pinch zoom (touch): the picture drawn g_zoom times its fitted size (g_fit),
 * cut to g_fit, its overflow shifted by g_zpx/g_zpy (0..1). g_dst is the
 * zoomed rectangle, so touches map to the game through it unchanged. */
#define ZOOM_MAX 4.0f
static SDL_Rect g_fit;
static float g_zoom = 1.0f, g_zpx = 0.5f, g_zpy = 0.5f;
void hle_screen_zoom_reset(void);
/* GL mode (hle_screen_gl set before hle_screen_open): the window is made for
 * OpenGL (1) or Vulkan (2) and has no SDL renderer; a host's 3D layer (a Direct3D 8 on GL) draws
 * into it and says where its picture is (hle_screen_gl_picture), so the
 * mouse maps as it does onto a scaled DirectDraw frame. */
int hle_screen_gl;
SDL_Window *hle_screen_sdl_window(void) { return g_win; }
static int g_src_w, g_src_h;                            /* the picture last shown: what the window maps to */
void hle_screen_gl_picture(int src_w, int src_h, int x, int y, int w, int h) {
    g_src_w = src_w, g_src_h = src_h;
    g_dst.x = x, g_dst.y = y, g_dst.w = w, g_dst.h = h;
}
static pthread_t g_video_thread;                         /* SDL's video calls stay on the thread that opened it */
static SDL_Texture *g_pre, *g_small;                     /* sharp's integer prescale; the bars' blur */
static int g_pre_w, g_pre_h;

/* How the picture is scaled, as the Windows presenter offers (F12 cycles):
 * sharp-bilinear (an integer prescale, nearest, then the rest bilinear),
 * smooth (bilinear), crt (sharp with scanlines), nearest, integer (whole
 * multiples only). The bars beside a letterboxed picture are a dark blur of
 * it, or black. */
static const char *const k_modes[] = { "sharp", "smooth", "crt", "nearest", "integer" };
static int g_mode, g_bars = 1;
void (*hle_screen_settings_hook)(const char *scale, int bars, int fullscreen);   /* to remember them */
int hle_screen_scale(const char *name) {
    for (int i = 0; i < 5; i++) if (name && !strcmp(name, k_modes[i])) return g_mode = i, 1;
    return 0;
}
void hle_screen_bars(int blur) { g_bars = blur != 0; }
static void settings_changed(void) {
    g_shown_seq = ~0u;
    if (hle_screen_settings_hook) hle_screen_settings_hook(k_modes[g_mode], g_bars, g_fullscreen);
}
static void menu_draw(void);
static int g_menu_depth;                                 /* hle_screen_menu's levels open; 0: closed */
static void set_fullscreen(int on) {
    g_fullscreen = on, g_menu_depth = 0;
#ifdef __EMSCRIPTEN__
    (void)on;                                              /* the page does it, on the key itself (a gesture) */
#else
    if (g_win) SDL_SetWindowFullscreen(g_win, on ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
#endif
    settings_changed();
}
void hle_screen_fullscreen(int on) { set_fullscreen(on != 0); }
const char *hle_screen_settings(int *bars, int *fullscreen) {
    if (bars) *bars = g_bars;
    if (fullscreen) *fullscreen = g_fullscreen;
    return k_modes[g_mode];
}
static uint32_t *g_conv;                                 /* 8-bit frames converted */
static size_t g_conv_n;
void (*hle_screen_frame_hook)(const uint8_t *px, int w, int h, int pitch, int bpp);   /* a recorder, a checksum */
/* A host's own picture at twice the size (an HD layer): fills out with 2w x
 * 2h XRGB from the 16-bit frame and returns 1, or returns 0 for the 1x. */
int (*hle_screen_compose_hook)(const uint8_t *px16, int pitch, int w, int h, uint32_t *out);
static uint32_t *g_hd;
static size_t g_hd_n;
/* The HD overlay (hle_screen_overlay_hook): a 2x ARGB layer blended over the
 * scaled 1x picture, so the CPU does not upscale the whole frame. */
int (*hle_screen_overlay_hook)(const uint8_t *px16, int pitch, int w, int h,
                               uint32_t *ov, int *dx, int *dy, int *dw, int *dh);
static uint32_t *g_ov;
static size_t g_ov_n;
static SDL_Texture *g_ov_tex;
static int g_ov_tw, g_ov_th;

static int vk_of(SDL_Scancode s) {
    if (s >= SDL_SCANCODE_A && s <= SDL_SCANCODE_Z) return 'A' + (s - SDL_SCANCODE_A);
    if (s >= SDL_SCANCODE_1 && s <= SDL_SCANCODE_9) return '1' + (s - SDL_SCANCODE_1);
    if (s >= SDL_SCANCODE_F1 && s <= SDL_SCANCODE_F12) return 0x70 + (s - SDL_SCANCODE_F1);
    if (s >= SDL_SCANCODE_KP_1 && s <= SDL_SCANCODE_KP_9) return 0x61 + (s - SDL_SCANCODE_KP_1);
    switch (s) {
    case SDL_SCANCODE_0: return '0';
    case SDL_SCANCODE_KP_0: return 0x60;
    case SDL_SCANCODE_RETURN: case SDL_SCANCODE_KP_ENTER: return 0x0D;
    case SDL_SCANCODE_ESCAPE: return 0x1B;
    case SDL_SCANCODE_BACKSPACE: return 0x08;
    case SDL_SCANCODE_TAB: return 0x09;
    case SDL_SCANCODE_SPACE: return 0x20;
    case SDL_SCANCODE_LSHIFT: case SDL_SCANCODE_RSHIFT: return 0x10;
    case SDL_SCANCODE_LCTRL: case SDL_SCANCODE_RCTRL: return 0x11;
    case SDL_SCANCODE_LALT: case SDL_SCANCODE_RALT: return 0x12;
    case SDL_SCANCODE_PAUSE: return 0x13;
    case SDL_SCANCODE_CAPSLOCK: return 0x14;
    case SDL_SCANCODE_PAGEUP: return 0x21;
    case SDL_SCANCODE_PAGEDOWN: return 0x22;
    case SDL_SCANCODE_END: return 0x23;
    case SDL_SCANCODE_HOME: return 0x24;
    case SDL_SCANCODE_LEFT: return 0x25;
    case SDL_SCANCODE_UP: return 0x26;
    case SDL_SCANCODE_RIGHT: return 0x27;
    case SDL_SCANCODE_DOWN: return 0x28;
    case SDL_SCANCODE_INSERT: return 0x2D;
    case SDL_SCANCODE_DELETE: return 0x2E;
    case SDL_SCANCODE_KP_MULTIPLY: return 0x6A;
    case SDL_SCANCODE_KP_PLUS: return 0x6B;
    case SDL_SCANCODE_KP_MINUS: return 0x6D;
    case SDL_SCANCODE_KP_PERIOD: return 0x6E;
    case SDL_SCANCODE_KP_DIVIDE: return 0x6F;
    case SDL_SCANCODE_SEMICOLON: return 0xBA;
    case SDL_SCANCODE_EQUALS: return 0xBB;
    case SDL_SCANCODE_COMMA: return 0xBC;
    case SDL_SCANCODE_MINUS: return 0xBD;
    case SDL_SCANCODE_PERIOD: return 0xBE;
    case SDL_SCANCODE_SLASH: return 0xBF;
    case SDL_SCANCODE_GRAVE: return 0xC0;
    case SDL_SCANCODE_LEFTBRACKET: return 0xDB;
    case SDL_SCANCODE_BACKSLASH: return 0xDC;
    case SDL_SCANCODE_RIGHTBRACKET: return 0xDD;
    case SDL_SCANCODE_APOSTROPHE: return 0xDE;
    default: return 0;
    }
}
/* the PC/AT set-1 scan code, which lParam carries */
static uint32_t scan_of(SDL_Scancode s) {
    static const uint8_t t[SDL_NUM_SCANCODES] = {
        [SDL_SCANCODE_ESCAPE] = 0x01, [SDL_SCANCODE_1] = 2, [SDL_SCANCODE_2] = 3, [SDL_SCANCODE_3] = 4, [SDL_SCANCODE_4] = 5,
        [SDL_SCANCODE_5] = 6, [SDL_SCANCODE_6] = 7, [SDL_SCANCODE_7] = 8, [SDL_SCANCODE_8] = 9, [SDL_SCANCODE_9] = 10,
        [SDL_SCANCODE_0] = 11, [SDL_SCANCODE_MINUS] = 12, [SDL_SCANCODE_EQUALS] = 13, [SDL_SCANCODE_BACKSPACE] = 14,
        [SDL_SCANCODE_TAB] = 15, [SDL_SCANCODE_Q] = 16, [SDL_SCANCODE_W] = 17, [SDL_SCANCODE_E] = 18, [SDL_SCANCODE_R] = 19,
        [SDL_SCANCODE_T] = 20, [SDL_SCANCODE_Y] = 21, [SDL_SCANCODE_U] = 22, [SDL_SCANCODE_I] = 23, [SDL_SCANCODE_O] = 24,
        [SDL_SCANCODE_P] = 25, [SDL_SCANCODE_LEFTBRACKET] = 26, [SDL_SCANCODE_RIGHTBRACKET] = 27, [SDL_SCANCODE_RETURN] = 28,
        [SDL_SCANCODE_LCTRL] = 29, [SDL_SCANCODE_A] = 30, [SDL_SCANCODE_S] = 31, [SDL_SCANCODE_D] = 32, [SDL_SCANCODE_F] = 33,
        [SDL_SCANCODE_G] = 34, [SDL_SCANCODE_H] = 35, [SDL_SCANCODE_J] = 36, [SDL_SCANCODE_K] = 37, [SDL_SCANCODE_L] = 38,
        [SDL_SCANCODE_SEMICOLON] = 39, [SDL_SCANCODE_APOSTROPHE] = 40, [SDL_SCANCODE_GRAVE] = 41, [SDL_SCANCODE_LSHIFT] = 42,
        [SDL_SCANCODE_BACKSLASH] = 43, [SDL_SCANCODE_Z] = 44, [SDL_SCANCODE_X] = 45, [SDL_SCANCODE_C] = 46, [SDL_SCANCODE_V] = 47,
        [SDL_SCANCODE_B] = 48, [SDL_SCANCODE_N] = 49, [SDL_SCANCODE_M] = 50, [SDL_SCANCODE_COMMA] = 51, [SDL_SCANCODE_PERIOD] = 52,
        [SDL_SCANCODE_SLASH] = 53, [SDL_SCANCODE_RSHIFT] = 54, [SDL_SCANCODE_LALT] = 56, [SDL_SCANCODE_SPACE] = 57,
        [SDL_SCANCODE_CAPSLOCK] = 58, [SDL_SCANCODE_F1] = 59, [SDL_SCANCODE_F2] = 60, [SDL_SCANCODE_F3] = 61, [SDL_SCANCODE_F4] = 62,
        [SDL_SCANCODE_F5] = 63, [SDL_SCANCODE_F6] = 64, [SDL_SCANCODE_F7] = 65, [SDL_SCANCODE_F8] = 66, [SDL_SCANCODE_F9] = 67,
        [SDL_SCANCODE_F10] = 68, [SDL_SCANCODE_F11] = 87, [SDL_SCANCODE_F12] = 88, [SDL_SCANCODE_HOME] = 71, [SDL_SCANCODE_UP] = 72,
        [SDL_SCANCODE_PAGEUP] = 73, [SDL_SCANCODE_LEFT] = 75, [SDL_SCANCODE_RIGHT] = 77, [SDL_SCANCODE_END] = 79,
        [SDL_SCANCODE_DOWN] = 80, [SDL_SCANCODE_PAGEDOWN] = 81, [SDL_SCANCODE_INSERT] = 82, [SDL_SCANCODE_DELETE] = 83 };
    return (unsigned)s < SDL_NUM_SCANCODES ? t[s] : 0;
}

/* window points -> game pixels, clamped to the picture */
static void to_game(int x, int y, int *gx, int *gy) {
    int w = g_src_w, h = g_src_h, bpp, pw, ph, ww, wh;
    if (w <= 0 || h <= 0) hle_dd_mode(&w, &h, &bpp);
    if (g_ren && g_win) {                                /* HiDPI: the renderer counts pixels, events points */
        SDL_GetRendererOutputSize(g_ren, &pw, &ph);
        SDL_GetWindowSize(g_win, &ww, &wh);
        if (ww > 0 && wh > 0) x = x * pw / ww, y = y * ph / wh;
    }
    else if (hle_screen_gl && g_win) {                  /* GL mode: the drawable counts pixels */
        if (hle_screen_gl == 2) SDL_Vulkan_GetDrawableSize(g_win, &pw, &ph);
        else SDL_GL_GetDrawableSize(g_win, &pw, &ph);
        SDL_GetWindowSize(g_win, &ww, &wh);
        if (ww > 0 && wh > 0) x = x * pw / ww, y = y * ph / wh;
    }
    if (g_dst.w <= 0 || g_dst.h <= 0) { *gx = x, *gy = y; return; }
    *gx = (int)((long long)(x - g_dst.x) * w / g_dst.w);
    *gy = (int)((long long)(y - g_dst.y) * h / g_dst.h);
    if (*gx < 0) *gx = 0;
    if (*gy < 0) *gy = 0;
    if (*gx >= w) *gx = w - 1;
    if (*gy >= h) *gy = h - 1;
}

#ifdef __EMSCRIPTEN__
static void web_present(const uint16_t *fb, int w, int h, int dx, int dy, int dw, int dh, int mode, int cw, int ch) {
    MAIN_THREAD_ASYNC_EM_ASM({ if (Module['ra2Present']) Module['ra2Present']($0, $1, $2, $3, $4, $5, $6, $7, $8, $9); },
                             fb, w, h, dx, dy, dw, dh, mode, cw, ch);
}
static void web_menu(int open) {
    static int was = -1;
    if (open == was) return;
    was = open;
    MAIN_THREAD_ASYNC_EM_ASM({ if (Module['ra2Menu']) Module['ra2Menu']($0); }, open);
}
#endif
/* Frame pacing, for flips: each frame stays on screen for the same number of
 * the display's refreshes. A game making 70 frames a second on a 175 Hz
 * display would otherwise show them for 2 or 3 refreshes in turn, which
 * judders; held to 3 it moves evenly at 58.3. The number is the fewest the
 * game's own time a frame fits in (10% over is let go, as an occasional
 * late frame; gaps of a tenth of a second and more are loads, not frames),
 * taken up at once and down only after a second of fitting, and never more
 * than a twentieth of a second. It is never fewer than keeps the game near 60
 * frames a second (HLE_MAX_FPS): a game of 1999 flips as fast as it can, as
 * its monitor's refresh let it on Windows, and on a 120 Hz display it would
 * otherwise draw twice the frames for the same picture. So 2 refreshes on a
 * 120 Hz display, 3 on a 175, 1 on a 60, 75 or 90.
 * The wait gives the machine up, so the game's other threads run; it sleeps
 * to within 0.2 ms of the refresh and spins only the rest (it slept whole
 * milliseconds and spun up to one, a sixteenth of the CPU at 60 a second). */
static uint64_t g_paced_back;                              /* when the last paced frame went out */
static int g_pace_k = 1, g_pace_hz;                         /* each frame held for k of the display's hz */
static void pace(void) {
    static uint64_t last_call;
    static double ema;
    static int lower;
    int k = g_pace_k;
    SDL_DisplayMode dm;
    int hz = SDL_GetCurrentDisplayMode(SDL_GetWindowDisplayIndex(g_win), &dm) == 0 && dm.refresh_rate > 0 ? dm.refresh_rate : 60;
    double freq = (double)SDL_GetPerformanceFrequency(), period = 1.0 / hz;
    uint64_t now = SDL_GetPerformanceCounter();
    if (g_paced_back) {
        double work = (double)(now - g_paced_back) / freq;  /* the game's time since the last frame went out */
        if (work < 0.1) ema = ema > 0 ? 0.9 * ema + 0.1 * work : work;   /* longer: a load, not a frame */
        int want = (int)ceil(ema / period - 0.1), most = hz / 20 > 1 ? hz / 20 : 1;   /* never under 20 a second */
        static int max_fps = -1;
        if (max_fps < 0) { const char *e = getenv("HLE_MAX_FPS"); max_fps = e ? atoi(e) : 60; }
        int least = max_fps > 0 ? (int)(hz / (double)max_fps + 0.25) : 1;
        if (least < 1) least = 1;
        if (want < least) want = least;
        if (want > most) want = most;
        if (want > k) k = want, lower = 0;
        else if (want < k && ++lower * k >= hz) k = want, lower = 0;  /* a second of frames that fit */
        else if (want >= k) lower = 0;
        uint64_t due = last_call + (uint64_t)(k * period * freq);
        if (now < due && due - now < (uint64_t)freq) {
            hle_block_begin();
            uint64_t spin = (uint64_t)(0.0002 * freq);   /* sleep to 0.2 ms short, then spin: sleeps overshoot */
            if (due - now > spin) {
                double s = (double)(due - now - spin) / freq;
                struct timespec ts = { (time_t)s, (long)((s - (double)(time_t)s) * 1e9) };
                nanosleep(&ts, NULL);
            }
            while (SDL_GetPerformanceCounter() < due) {}
            hle_block_end();
            now = due;
        }
    }
    last_call = now;
    g_pace_k = k, g_pace_hz = hz;
}
/* The least time between two presents of a game that does not flip (ms):
 * HLE_PRESENT_MS, 15 by default (about 60 a second). A slow device spends
 * the present's time off the game's logic, so fewer presents run it faster. */
static uint32_t present_ms(void) {
    static int ms = -1;
    if (ms < 0) { const char *e = SDL_getenv("HLE_PRESENT_MS"); ms = e && atoi(e) > 0 ? atoi(e) : 15; }
    return (uint32_t)ms;
}
static void touch_draw(void);
/* The present's parts, timed while hle_shim_timing (--profile): 0 the frame
 * hook, 1 overlay/compose, 2 texture upload, 3 draw, 4 RenderPresent,
 * 5 PumpEvents. hle_present_n[0] counts presents, [5] event pumps. */
uint64_t hle_present_ns[8];
uint32_t hle_present_n[8];
static uint64_t g_pt;
#define PT_START() do { if (hle_shim_timing) g_pt = hle_ticks(); } while (0)
#define PT(k) do { if (hle_shim_timing) { uint64_t t_ = hle_ticks(); hle_present_ns[k] += hle_ticks_ns(t_ - g_pt); g_pt = t_; } } while (0)
/* flip: the game flipped, and this frame is shown, paced (pace). */
static void present(int flip) {
    const uint8_t *px;
    const uint32_t *pal;
    int w, h, pitch, bpp;
    unsigned seq;
    pal = NULL;
    if (!hle_dd_frame(&px, &w, &h, &pitch, &bpp, &pal, &seq) && !hle_gdi_window_frame(&px, &w, &h, &pitch, &bpp, &seq)) return;
    uint32_t now = hle_ticks_ms();
    g_src_w = w, g_src_h = h;
    if (!flip && seq == g_shown_seq && now - g_last_present < 250) return;
    if (!flip && now - g_last_present < present_ms()) return;
    g_shown_seq = seq, g_last_present = now;
    PT_START();
    hle_present_n[0]++;
    if (hle_screen_frame_hook) hle_screen_frame_hook(px, w, h, pitch, bpp);
    if (!g_ren) return;
#ifdef __EMSCRIPTEN__
    /* The page draws the frame with WebGL on its own thread (index.html's
     * ra2Present): SDL's software renderer scaled it on this thread and
     * handed the whole window's pixels over, which took longer than the
     * game's frame, so the game never went idle. The frame goes as RGB565
     * in one of two buffers (the page reads it later); the place in the
     * window is this thread's, so the mouse maps the same way. SDL still
     * draws the host's menu (F10), over which the page's canvas hides. */
    if (bpp == 16) {
        int ww, wh;
        SDL_GetRendererOutputSize(g_ren, &ww, &wh);
        if ((long long)ww * h > (long long)wh * w) g_dst.h = wh, g_dst.w = (int)((long long)wh * w / h);
        else g_dst.w = ww, g_dst.h = (int)((long long)ww * h / w);
        if (g_mode == 4) {
            int k = ww / w < wh / h ? ww / w : wh / h;
            if (k < 1) k = 1;
            g_dst.w = w * k, g_dst.h = h * k;
        }
        g_dst.x = (ww - g_dst.w) / 2, g_dst.y = (wh - g_dst.h) / 2;
        if (g_menu_depth) {                              /* SDL's picture and menu: the page's canvas steps aside */
            web_menu(1);
        } else {
            static uint16_t *fb[2];
            static size_t fbn;
            static int which;
            if (fbn < (size_t)w * h) {
                free(fb[0]); free(fb[1]);
                fbn = (size_t)w * h;
                fb[0] = (uint16_t *)malloc(fbn * 2), fb[1] = (uint16_t *)malloc(fbn * 2);
            }
            which ^= 1;
            uint16_t *dst = fb[which];
            if (dst) {
                for (int y = 0; y < h; y++) memcpy(dst + (size_t)y * w, px + (size_t)y * pitch, (size_t)w * 2);
                web_menu(0);
                web_present(dst, w, h, g_dst.x, g_dst.y, g_dst.w, g_dst.h, g_mode, ww, wh);
            }
            return;
        }
    }
#endif
    PT(0);
    int hd = 0, ov = 0, ovx = 0, ovy = 0, ovw = 0, ovh = 0;
    if (bpp == 16 && hle_screen_overlay_hook) {        /* 1x base scaled on the GPU + an Nx overlay */
        if (g_ov_n < (size_t)w * h * 9) {              /* sized for the largest scale (3x) */
            free(g_ov); g_ov_n = (size_t)w * h * 9; g_ov = (uint32_t *)calloc(g_ov_n, 4);
            if (g_ov_tex) { SDL_DestroyTexture(g_ov_tex); g_ov_tex = NULL; }
        }
        ov = g_ov ? hle_screen_overlay_hook(px, pitch, w, h, g_ov, &ovx, &ovy, &ovw, &ovh) : 0;
    }
    if (!ov && bpp == 16 && hle_screen_compose_hook) { /* the old full 2x compose (other titles) */
        if (g_hd_n < (size_t)w * h * 4) { free(g_hd); g_hd_n = (size_t)w * h * 4; g_hd = (uint32_t *)malloc(g_hd_n * 4); }
        hd = g_hd && hle_screen_compose_hook(px, pitch, w, h, g_hd);
    }
    PT(1);
    int fmt = bpp == 16 && !hd ? SDL_PIXELFORMAT_RGB565 : SDL_PIXELFORMAT_RGB888;
    int tw = hd ? 2 * w : w, th = hd ? 2 * h : h;
    if (!g_tex || g_tex_w != tw || g_tex_h != th || g_tex_fmt != fmt) {
        if (g_tex) SDL_DestroyTexture(g_tex);
        g_tex = SDL_CreateTexture(g_ren, (Uint32)fmt, SDL_TEXTUREACCESS_STREAMING, tw, th);
        g_tex_w = tw, g_tex_h = th, g_tex_fmt = fmt;
        {   /* the format the renderer really keeps: another one costs a CPU conversion each frame */
            SDL_RendererInfo ri;
            int native = 0;
            if (SDL_GetRendererInfo(g_ren, &ri) == 0)
                for (Uint32 i = 0; i < ri.num_texture_formats; i++) native |= ri.texture_formats[i] == (Uint32)fmt;
            fprintf(stderr, "[screen] frame texture %dx%d %s%s\n", tw, th, SDL_GetPixelFormatName((Uint32)fmt),
                    !native ? " (not native: SDL converts it)" : "");
        }
    }
    if (hd) {
        SDL_UpdateTexture(g_tex, NULL, g_hd, tw * 4);
    } else if (bpp == 8) {
        if (g_conv_n < (size_t)w * h) { free(g_conv); g_conv_n = (size_t)w * h; g_conv = (uint32_t *)malloc(g_conv_n * 4); }
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                uint32_t e = pal ? pal[px[(size_t)y * pitch + x]] : 0;   /* PALETTEENTRY r, g, b */
                g_conv[(size_t)y * w + x] = 0xFF000000u | (e & 0xFF) << 16 | (e & 0xFF00) | (e >> 16 & 0xFF);
            }
        SDL_UpdateTexture(g_tex, NULL, g_conv, w * 4);
    } else {
        SDL_UpdateTexture(g_tex, NULL, px, pitch);
    }
    PT(2);
    int ww, wh;
    SDL_GetRendererOutputSize(g_ren, &ww, &wh);
    if ((long long)ww * h > (long long)wh * w) g_dst.h = wh, g_dst.w = (int)((long long)wh * w / h);
    else g_dst.w = ww, g_dst.h = (int)((long long)ww * h / w);
    if (g_mode == 4) {                                   /* integer: whole multiples of the game's size */
        int k = ww / w < wh / h ? ww / w : wh / h;
        if (k < 1) k = 1;
        g_dst.w = w * k, g_dst.h = h * k;
    }
    {   /* HLE_SIZE: "stretch" fills the window (the shape lost); "1".."4" that
         * many times the game's size, or the fit when it would not fit; else
         * (unset, "fit") the fit above. The filter (g_mode) stays apart. */
        static int size = -2;                            /* -1 stretch, 0 fit, N times */
        if (size == -2) {
            const char *e = SDL_getenv("HLE_SIZE");
            size = !e ? 0 : !strcmp(e, "stretch") ? -1 : atoi(e) > 0 && atoi(e) <= 8 ? atoi(e) : 0;
        }
        if (size < 0) g_dst.w = ww, g_dst.h = wh;
        else if (size > 0 && w * size <= ww && h * size <= wh) g_dst.w = w * size, g_dst.h = h * size;
    }
    g_dst.x = (ww - g_dst.w) / 2, g_dst.y = (wh - g_dst.h) / 2;
    g_fit = g_dst;
    if (g_zoom > 1.0f) {                                 /* pinch zoom: larger, shifted, cut to the fit below */
        int zw = (int)(g_fit.w * g_zoom), zh = (int)(g_fit.h * g_zoom);
        g_dst.w = zw, g_dst.h = zh;
        g_dst.x = g_fit.x - (int)((zw - g_fit.w) * g_zpx), g_dst.y = g_fit.y - (int)((zh - g_fit.h) * g_zpy);
    }
    SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 255);
    SDL_RenderClear(g_ren);
    if (g_bars && (g_fit.w < ww || g_fit.h < wh)) {     /* the bars: the picture, tiny then stretched, dark */
        if (!g_small) g_small = SDL_CreateTexture(g_ren, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_TARGET, 24, 15);
        if (g_small) {
            SDL_SetTextureScaleMode(g_tex, SDL_ScaleModeLinear);
            SDL_SetRenderTarget(g_ren, g_small);
            SDL_RenderCopy(g_ren, g_tex, NULL, NULL);
            SDL_SetRenderTarget(g_ren, NULL);
            SDL_SetTextureScaleMode(g_small, SDL_ScaleModeLinear);
            SDL_SetTextureColorMod(g_small, 90, 90, 90);
            SDL_RenderCopy(g_ren, g_small, NULL, NULL);
        }
    }
    if (g_zoom > 1.0f) SDL_RenderSetClipRect(g_ren, &g_fit);
    int sharp = g_mode == 0 || g_mode == 2;
    /* the prescale from the fitted size: zoomed, it would outgrow a phone's textures */
    int k = g_fit.w / tw < g_fit.h / th ? g_fit.w / tw : g_fit.h / th;
    if (sharp && k >= 1 && (tw * k != g_dst.w || th * k != g_dst.h)) {
        if (!g_pre || g_pre_w != tw * k || g_pre_h != th * k) {
            if (g_pre) SDL_DestroyTexture(g_pre);
            g_pre = SDL_CreateTexture(g_ren, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_TARGET, tw * k, th * k);
            g_pre_w = tw * k, g_pre_h = th * k;
        }
    }
    if (sharp && k >= 1 && g_pre && (tw * k != g_dst.w || th * k != g_dst.h)) {
        SDL_SetTextureScaleMode(g_tex, SDL_ScaleModeNearest);
        SDL_SetRenderTarget(g_ren, g_pre);
        SDL_RenderCopy(g_ren, g_tex, NULL, NULL);
        SDL_SetRenderTarget(g_ren, NULL);
        SDL_SetTextureScaleMode(g_pre, SDL_ScaleModeLinear);
        SDL_RenderCopy(g_ren, g_pre, NULL, &g_dst);
    } else {
        SDL_SetTextureScaleMode(g_tex, g_mode == 1 ? SDL_ScaleModeLinear : SDL_ScaleModeNearest);
        SDL_RenderCopy(g_ren, g_tex, NULL, &g_dst);
    }
    if (ov) {                                            /* the Nx HD overlay (ov = N), over the scaled 1x picture */
        int ow = ov * w, oh = ov * h;
        if (!g_ov_tex || g_ov_tw != ow || g_ov_th != oh) {
            if (g_ov_tex) SDL_DestroyTexture(g_ov_tex);
            g_ov_tex = SDL_CreateTexture(g_ren, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, ow, oh);
            g_ov_tw = ow, g_ov_th = oh;
            if (g_ov_tex) SDL_SetTextureBlendMode(g_ov_tex, SDL_BLENDMODE_BLEND);
            ovx = 0, ovy = 0, ovw = ow, ovh = oh;        /* a new texture: upload all of it */
        }
        if (g_ov_tex && ovw > 0 && ovh > 0) {
            SDL_Rect ur = { ovx, ovy, ovw, ovh };
            SDL_UpdateTexture(g_ov_tex, &ur, g_ov + (size_t)ovy * ow + ovx, ow * 4);
        }
        if (g_ov_tex) {
            SDL_SetTextureScaleMode(g_ov_tex, SDL_ScaleModeNearest);
            SDL_RenderCopy(g_ren, g_ov_tex, NULL, &g_dst);
        }
    }
    if (g_mode == 2 && g_dst.h >= 2 * h) {                /* crt: the lower third of every game row darker */
        SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 96);
        for (int y = 0; y < h; y++) {
            int y0 = g_dst.y + (int)((long long)(3 * y + 2) * g_dst.h / (3 * h)), y1 = g_dst.y + (int)((long long)(y + 1) * g_dst.h / h);
            SDL_Rect r = { g_dst.x, y0, g_dst.w, y1 > y0 ? y1 - y0 : 1 };
            SDL_RenderFillRect(g_ren, &r);
        }
        SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_NONE);
    }
    SDL_RenderSetClipRect(g_ren, NULL);
    touch_draw();
    if (g_menu_depth) menu_draw();
    PT(3);
    if (flip) pace();
    SDL_RenderPresent(g_ren);
    PT(4);
    if (flip) g_paced_back = SDL_GetPerformanceCounter();
    static int trace = -1;
    static uint32_t t0, prev, shown, flips, worst, late;
    if (trace < 0) trace = getenv("HLE_PRESENTTRACE") != NULL;
    if (trace) {                                         /* HLE_PRESENTTRACE: frames shown a second, and the longest wait */
        uint32_t t = hle_ticks_ms();
        if (prev && t - prev > worst) worst = t - prev;
        if (prev && flip && g_pace_hz && (t - prev) * (uint32_t)g_pace_hz > 1500u * (uint32_t)g_pace_k) late++;   /* held a refresh or more too long */
        prev = t, shown++, flips += flip != 0;
        if (!t0) t0 = t;
        else if (t - t0 >= 5000) {
            fprintf(stderr, "[screen] %.1f frames shown/s (%u of them flips), longest gap %u ms; flips held %d of %d Hz, %u late\n",
                    shown * 1000.0 / (t - t0), flips, worst, g_pace_k, g_pace_hz, late);
            t0 = t, shown = flips = worst = late = 0;
        }
    }
}

/* Who shows the cursor. A DirectDraw game draws its own into the picture, so
 * by default the window has none. A game whose menus are Win32 dialogs (Red
 * Alert 2) uses the Windows cursor there: its host's hook says when, and the
 * window shows the cursor the game set, built from its resources. */
int (*hle_screen_cursor_hook)(void);
int (*hle_screen_key_hook)(int sdl_scancode, int down);
void (*hle_screen_poll_hook)(void);

/* The window's title, for a host that shows what is playing (a mod). */
void hle_screen_title(const char *title) { if (g_win) SDL_SetWindowTitle(g_win, title); }

static SDL_Cursor *sdl_cursor_from(const hle_cursor_image_t *img) {
    /* Windows' AND/XOR masks to SDL's mask/data: AND 0 is opaque (XOR 0 black,
     * 1 white); AND 1 is transparent (XOR 1 inverts). */
    uint8_t data[128], mask[128];
    for (int i = 0; i < 128; i++) {
        mask[i] = (uint8_t)~img->and_mask[i];
        data[i] = (uint8_t)((~img->xor_mask[i] & mask[i]) | (img->xor_mask[i] & img->and_mask[i]));
    }
    return SDL_CreateCursor(data, mask, 32, 32, img->hot_x, img->hot_y);
}

static void update_cursor(void) {
    static int was_shown = 0;
    static uint32_t was_handle;
    static SDL_Cursor *made;
    int show = g_menu_depth || (hle_screen_cursor_hook && hle_screen_cursor_hook());
    uint32_t handle = g_menu_depth ? 0 : hle_current_cursor();   /* over a menu, the system arrow */
    if (show == was_shown && (!show || handle == was_handle)) return;
    if (show && handle != was_handle) {
        const hle_cursor_image_t *img = hle_cursor_image(handle);
        SDL_Cursor *c = img ? sdl_cursor_from(img) : NULL;
        SDL_SetCursor(c ? c : SDL_GetDefaultCursor());
        if (made) SDL_FreeCursor(made);
        made = c;
        was_handle = handle;
    }
    if (show != was_shown) SDL_ShowCursor(show ? SDL_ENABLE : SDL_DISABLE);
    fprintf(stderr, "[screen] cursor: %s\n", !show ? "drawn by the game" : made ? "the game's, from its resources" : "the system arrow");
    was_shown = show;
}

/* ---- the popup menu (hle_screen_menu) ----
 * Drawn over the picture in the renderer's pixels, in a desktop menu's look:
 * light rows, a blue highlight, a check mark, a submenu's arrow. */
#define MENU_LEVELS 4
static struct { const hle_menu_item_t *items; int n, x, y, w, h, sel; } g_menu[MENU_LEVELS];
static void (*g_menu_chosen)(int id);
static TTF_Font *g_menu_font;
static int g_menu_scale, g_menu_eat_up;                  /* renderer pixels per window point; the closing click's release */

static int menu_ok(const hle_menu_item_t *it) { return it->label && (it->sub || it->id); }
static int menu_item_h(const hle_menu_item_t *it) {
    return it->label ? TTF_FontHeight(g_menu_font) + 8 * g_menu_scale : 7 * g_menu_scale;
}
static int menu_text_w(const char *s) {
    int w = 0, h;
    if (s && *s) TTF_SizeUTF8(g_menu_font, s, &w, &h);
    return w;
}
static void menu_dirty(void) { g_shown_seq = ~0u; }
static void menu_close(void) { g_menu_depth = 0; menu_dirty(); }

/* Level l at (x, y), kept inside the window: a submenu that would leave it
 * on the right opens to its parent's left instead. */
static void menu_place(int l, int x, int y) {
    int pw, ph, s = g_menu_scale, w = 0, h = 2 * s;
    SDL_GetRendererOutputSize(g_ren, &pw, &ph);
    for (int i = 0; i < g_menu[l].n; i++) {
        const hle_menu_item_t *it = &g_menu[l].items[i];
        int iw = menu_text_w(it->label) + (it->hint ? menu_text_w(it->hint) + 32 * s : 0);
        if (iw > w) w = iw;
        h += menu_item_h(it);
    }
    w += 2 * 26 * s;                                     /* the check's column and the arrow's */
    if (x + w > pw) x = l ? g_menu[l - 1].x - w : pw - w;
    if (y + h > ph) y = ph - h;
    g_menu[l].x = x < 0 ? 0 : x, g_menu[l].y = y < 0 ? 0 : y, g_menu[l].w = w, g_menu[l].h = h, g_menu[l].sel = -1;
}
static int menu_row_y(int l, int row) {
    int y = g_menu[l].y + g_menu_scale;
    for (int i = 0; i < row; i++) y += menu_item_h(&g_menu[l].items[i]);
    return y;
}
static int menu_first(int l, int from, int step) {      /* the next item that can be chosen, or -1 */
    for (int k = 0, i = from; k < g_menu[l].n; k++) {
        i = (i + step + g_menu[l].n) % g_menu[l].n;
        if (menu_ok(&g_menu[l].items[i])) return i;
    }
    return -1;
}
/* Row `row` of level l is highlighted: its submenu opens, any deeper closes. */
static void menu_select(int l, int row) {
    const hle_menu_item_t *it = &g_menu[l].items[row];
    g_menu[l].sel = row, g_menu_depth = l + 1;
    if (it->sub && it->nsub > 0 && l + 1 < MENU_LEVELS) {
        g_menu[l + 1].items = it->sub, g_menu[l + 1].n = it->nsub;
        menu_place(l + 1, g_menu[l].x + g_menu[l].w - 2 * g_menu_scale, menu_row_y(l, row) - g_menu_scale);
        g_menu_depth = l + 2;
    }
    menu_dirty();
}
static void menu_choose(int l, int row) {
    const hle_menu_item_t *it = &g_menu[l].items[row];
    if (!menu_ok(it)) return;
    if (it->sub) {                                       /* into the submenu, its first item */
        menu_select(l, row);
        if (g_menu_depth == l + 2) g_menu[l + 1].sel = menu_first(l + 1, -1, 1);
        return;
    }
    void (*chosen)(int) = g_menu_chosen;
    menu_close();
    fprintf(stderr, "[screen] menu: %s\n", it->label);
    if (chosen) chosen(it->id);
}
static int menu_hit(int x, int y, int *row) {           /* the deepest level under (x, y), or -1 */
    for (int l = g_menu_depth - 1; l >= 0; l--) {
        if (x < g_menu[l].x || x >= g_menu[l].x + g_menu[l].w || y < g_menu[l].y || y >= g_menu[l].y + g_menu[l].h) continue;
        int ry = g_menu[l].y + g_menu_scale;
        *row = -1;
        for (int i = 0; i < g_menu[l].n && *row < 0; i++) {
            ry += menu_item_h(&g_menu[l].items[i]);
            if (y < ry) *row = i;
        }
        return l;
    }
    return -1;
}
static void menu_key(SDL_Scancode sc) {
    int l = g_menu_depth - 1, sel = g_menu[l].sel;
    if (sc == SDL_SCANCODE_ESCAPE) { if (l) g_menu_depth = l, menu_dirty(); else menu_close(); }
    else if (sc == SDL_SCANCODE_F10) menu_close();
    else if (sc == SDL_SCANCODE_UP || sc == SDL_SCANCODE_DOWN) {
        int to = menu_first(l, sel < 0 && sc == SDL_SCANCODE_UP ? 0 : sel, sc == SDL_SCANCODE_UP ? -1 : 1);
        if (to >= 0) g_menu[l].sel = to, g_menu_depth = l + 1, menu_dirty();
    }
    else if (sc == SDL_SCANCODE_LEFT && l) g_menu_depth = l, menu_dirty();
    else if (sel >= 0 && ((sc == SDL_SCANCODE_RIGHT && g_menu[l].items[sel].sub) ||
                          sc == SDL_SCANCODE_RETURN || sc == SDL_SCANCODE_KP_ENTER || sc == SDL_SCANCODE_SPACE))
        menu_choose(l, sel);
}
/* An event while the menu is open: the menu's, never the game's. */
static int menu_event(const SDL_Event *e) {
    int x, y, row = -1, l, pw, ph, ww, wh;
    switch (e->type) {
    case SDL_MOUSEMOTION: case SDL_MOUSEBUTTONDOWN: case SDL_MOUSEBUTTONUP:
        x = e->type == SDL_MOUSEMOTION ? e->motion.x : e->button.x;
        y = e->type == SDL_MOUSEMOTION ? e->motion.y : e->button.y;
        SDL_GetRendererOutputSize(g_ren, &pw, &ph);
        SDL_GetWindowSize(g_win, &ww, &wh);
        if (ww > 0 && wh > 0) x = x * pw / ww, y = y * ph / wh;
        l = menu_hit(x, y, &row);
        if (e->type == SDL_MOUSEMOTION) { if (l >= 0 && row >= 0 && g_menu[l].sel != row) menu_select(l, row); }
        else if (e->type == SDL_MOUSEBUTTONDOWN) {
            g_menu_eat_up = 1;                           /* a click that closes the menu: its release isn't the game's */
            if (l < 0) menu_close(); else if (row >= 0) menu_choose(l, row);
        } else g_menu_eat_up = 0;
        return 1;
    case SDL_KEYDOWN: menu_key(e->key.keysym.scancode); return 1;
    case SDL_KEYUP: case SDL_TEXTINPUT: case SDL_MOUSEWHEEL: return 1;
    default: return 0;
    }
}

static void menu_text(const char *s, int x, int y, SDL_Color c) {
    SDL_Surface *t = s && *s ? TTF_RenderUTF8_Blended(g_menu_font, s, c) : NULL;
    if (!t) return;
    SDL_Texture *tx = SDL_CreateTextureFromSurface(g_ren, t);   /* ponytail: made each frame; a cache if a menu ever grows long */
    SDL_Rect r = { x, y, t->w, t->h };
    if (tx) SDL_RenderCopy(g_ren, tx, NULL, &r), SDL_DestroyTexture(tx);
    SDL_FreeSurface(t);
}
static void menu_lines(const int *pts, int n, int dx, int dy) {   /* a stroke two pixels wide per scale */
    for (int t = 0; t < 2 * g_menu_scale; t++)
        for (int i = 0; i + 3 < 2 * n; i += 2)
            SDL_RenderDrawLine(g_ren, dx + pts[i] * g_menu_scale, dy + pts[i + 1] * g_menu_scale + t,
                               dx + pts[i + 2] * g_menu_scale, dy + pts[i + 3] * g_menu_scale + t);
}
static void menu_draw(void) {
    int s = g_menu_scale, fh = TTF_FontHeight(g_menu_font);
    for (int l = 0; l < g_menu_depth; l++) {
        SDL_Rect box = { g_menu[l].x, g_menu[l].y, g_menu[l].w, g_menu[l].h };
        SDL_SetRenderDrawColor(g_ren, 242, 242, 242, 255);
        SDL_RenderFillRect(g_ren, &box);
        SDL_SetRenderDrawColor(g_ren, 150, 150, 150, 255);
        SDL_RenderDrawRect(g_ren, &box);
        int y = g_menu[l].y + s;
        for (int i = 0; i < g_menu[l].n; i++) {
            const hle_menu_item_t *it = &g_menu[l].items[i];
            int h = menu_item_h(it), x = g_menu[l].x;
            if (!it->label) {
                SDL_SetRenderDrawColor(g_ren, 200, 200, 200, 255);
                SDL_Rect sep = { x + 8 * s, y + h / 2, g_menu[l].w - 16 * s, s };
                SDL_RenderFillRect(g_ren, &sep);
                y += h;
                continue;
            }
            int lit = i == g_menu[l].sel && menu_ok(it);
            SDL_Color c = lit ? (SDL_Color){ 255, 255, 255, 255 } : menu_ok(it) ? (SDL_Color){ 20, 20, 20, 255 }
                                                                               : (SDL_Color){ 150, 150, 150, 255 };
            if (lit) {
                SDL_Rect r = { x + 2 * s, y, g_menu[l].w - 4 * s, h };
                SDL_SetRenderDrawColor(g_ren, 0, 103, 192, 255);
                SDL_RenderFillRect(g_ren, &r);
            }
            SDL_SetRenderDrawColor(g_ren, c.r, c.g, c.b, 255);
            int mid = y + h / 2;
            if (it->checked) { static const int tick[] = { 9, 0, 12, 3, 18, -4 }; menu_lines(tick, 3, x, mid - s); }
            if (it->sub) { static const int arrow[] = { 0, -4, 4, 0, 0, 4 }; menu_lines(arrow, 3, x + g_menu[l].w - 16 * s, mid - s); }
            menu_text(it->label, x + 26 * s, y + (h - fh) / 2, c);
            if (it->hint) menu_text(it->hint, x + g_menu[l].w - 26 * s - menu_text_w(it->hint), y + (h - fh) / 2, c);
            y += h;
        }
    }
}

#ifndef __EMSCRIPTEN__
int hle_screen_menu_open(void) { return g_menu_depth > 0; }
#endif

/* The size of the display the window is on (the first, before it opens):
 * SDL's desktop mode, in points, so a Retina Mac's logical size. */
int hle_screen_desktop(int *w, int *h) {
    SDL_DisplayMode m;
    int i = g_win ? SDL_GetWindowDisplayIndex(g_win) : 0;
    if (!SDL_WasInit(SDL_INIT_VIDEO) || SDL_GetDesktopDisplayMode(i < 0 ? 0 : i, &m) != 0) return 0;
    *w = m.w, *h = m.h;
    return 1;
}
#ifdef __EMSCRIPTEN__
/* In a browser the page draws the menu (index.html's ra2PopupMenu), as HTML
 * at the mouse: SDL's own, drawn with its software renderer, took longer
 * than a frame to show, and the game then never went idle to read a key.
 * The items go over as JSON; the page writes the choice (0: none yet, -1:
 * closed) into g_web_choice, and the pump calls chosen() on this thread. */
static volatile int g_web_choice;
static void (*g_web_chosen)(int id);
static char g_web_menu_json[16384];
static size_t json_item(char *out, size_t n, const hle_menu_item_t *it) {
    size_t k = 0;
    if (!it->label) return (size_t)snprintf(out, n, "null");
    k += (size_t)snprintf(out + k, n - k, "{\"label\":\"");
    for (const char *p = it->label; *p && k + 8 < n; p++) {
        if (*p == '"' || *p == '\\') out[k++] = '\\';
        out[k++] = *p;
    }
    k += (size_t)snprintf(out + k, n - k, "\",\"hint\":\"%s\",\"id\":%d,\"checked\":%d", it->hint ? it->hint : "", it->id, it->checked ? 1 : 0);
    if (it->sub && it->nsub > 0) {
        k += (size_t)snprintf(out + k, n - k, ",\"sub\":[");
        for (int i = 0; i < it->nsub && k + 64 < n; i++) {
            if (i) out[k++] = ',';
            k += json_item(out + k, n - k, &it->sub[i]);
        }
        k += (size_t)snprintf(out + k, n - k, "]");
    }
    k += (size_t)snprintf(out + k, n - k, "}");
    return k;
}
void hle_screen_menu(const hle_menu_item_t *items, int n, void (*chosen)(int id)) {
    if (!items || n <= 0) return;
    size_t k = 0;
    g_web_menu_json[k++] = '[';
    for (int i = 0; i < n && k + 64 < sizeof g_web_menu_json; i++) {
        if (i) g_web_menu_json[k++] = ',';
        k += json_item(g_web_menu_json + k, sizeof g_web_menu_json - k, &items[i]);
    }
    g_web_menu_json[k++] = ']';
    g_web_menu_json[k] = 0;
    g_web_choice = 0, g_web_chosen = chosen;
    MAIN_THREAD_ASYNC_EM_ASM({ if (Module['ra2PopupMenu']) Module['ra2PopupMenu'](UTF8ToString($0), $1); else HEAP32[$1 >> 2] = -1; },
                             g_web_menu_json, &g_web_choice);
}
int hle_screen_menu_open(void) { return g_web_chosen != NULL; }
static void web_menu_poll(void) {
    if (!g_web_chosen || !g_web_choice) return;
    int id = g_web_choice;
    void (*cb)(int) = g_web_chosen;
    g_web_choice = 0, g_web_chosen = NULL;
    if (id > 0) cb(id);
}
#endif
#ifndef __EMSCRIPTEN__
void hle_screen_menu(const hle_menu_item_t *items, int n, void (*chosen)(int id)) {
    int pw, ph, ww, wh, mx, my;
    if (!g_ren || !items || n <= 0) return;
    SDL_GetRendererOutputSize(g_ren, &pw, &ph);
    SDL_GetWindowSize(g_win, &ww, &wh);
    int scale = ww > 0 && pw >= 2 * ww ? 2 : 1;          /* ponytail: 1x or 2x; a fractional HiDPI scale gets 1x */
    if (!g_menu_font || scale != g_menu_scale) {
        const char *path = TTF_WasInit() || TTF_Init() == 0 ? hle_font_file("MS Sans Serif", 0, 0) : NULL;
        if (g_menu_font) TTF_CloseFont(g_menu_font);
        g_menu_font = path ? TTF_OpenFont(path, 16 * scale) : NULL;
        g_menu_scale = scale;
        if (!g_menu_font) { fprintf(stderr, "[screen] no font for the menu: install fonts-liberation\n"); return; }
    }
    SDL_GetMouseState(&mx, &my);
    if (ww > 0 && wh > 0) mx = mx * pw / ww, my = my * ph / wh;
    g_menu[0].items = items, g_menu[0].n = n;
    menu_place(0, mx, my);
    g_menu_depth = 1, g_menu_chosen = chosen, g_menu_eat_up = 0;
    menu_dirty();
}
#endif

/* A rectangle of the renderer's pixels (g_dst) in the window's points, which
 * mouse calls take: they differ on a HiDPI display. */
static SDL_Rect to_points(SDL_Rect r) {
    int pw, ph, ww, wh;
    SDL_GetRendererOutputSize(g_ren, &pw, &ph);
    SDL_GetWindowSize(g_win, &ww, &wh);
    if (pw <= 0 || ph <= 0) return r;
    int x0 = (int)(((long long)r.x * ww + pw - 1) / pw), y0 = (int)(((long long)r.y * wh + ph - 1) / ph);
    int x1 = (int)((long long)(r.x + r.w) * ww / pw), y1 = (int)((long long)(r.y + r.h) * wh / ph);
    return (SDL_Rect){ x0, y0, x1 - x0, y1 - y0 };
}

/* The mouse is held inside the picture while the window has focus, as a
 * full-screen DirectDraw game's is held on the screen on Windows. The game
 * draws its own cursor, from its messages or DirectInput, and those stop at
 * the window's edge; the desktop's pointer would go on past it, and a click
 * there would land on another window. It is let go when the window loses
 * focus, while the popup menu is open, or on Ctrl+Cmd (Ctrl+Win, Ctrl+Super);
 * a click in the window takes it back. A host turns it off with
 * hle_screen_confine. */
#if defined(__APPLE__)
#define RELEASE_KEYS "Ctrl+Cmd"
#elif defined(_WIN32)
#define RELEASE_KEYS "Ctrl+Win"
#else
#define RELEASE_KEYS "Ctrl+Super"
#endif
static int g_confine = 1, g_released, g_regrab_up;
void hle_screen_confine(int on) { g_confine = on != 0; }
/* A game holding the mouse for its movement alone (dinput.c): while the
 * mouse is held, the window gives that (SDL's relative mode), so the game's
 * cursor goes on where the desktop's pointer would stop at the edge. */
static int g_want_rel, g_rel;
static double g_rel_fx, g_rel_fy;                          /* game pixels not yet whole */
static void want_relative(int on) { g_want_rel = on; }
int hle_screen_relative(void) { return g_rel; }
static void warp(int x, int y);
static void warp_to_cursor(void) { int x, y; hle_input_position(&x, &y); warp(x, y); }
static void confine_update(void) {
#if !defined(__EMSCRIPTEN__) && SDL_VERSION_ATLEAST(2, 0, 18)
    static int held;
    static SDL_Rect at;
    if (!g_win || !g_ren) return;
    int want = g_confine && !g_released && !g_menu_depth && g_dst.w > 0 && g_dst.h > 0
               && (SDL_GetWindowFlags(g_win) & SDL_WINDOW_INPUT_FOCUS);
    SDL_Rect r = want ? to_points(g_dst) : (SDL_Rect){ 0, 0, 0, 0 };
    int rel = want && g_want_rel;
    if (rel != g_rel) {
        SDL_SetRelativeMouseMode(rel ? SDL_TRUE : SDL_FALSE);
        g_rel = rel, g_rel_fx = g_rel_fy = 0;
        if (!rel && want) warp_to_cursor();              /* the desktop's pointer where the game's cursor is */
        fprintf(stderr, "[screen] mouse %s\n", rel ? "relative: the game has its movement" : "absolute");
    }
    if (want == held && (!want || !memcmp(&r, &at, sizeof r))) return;
    SDL_SetWindowMouseRect(g_win, want ? &r : NULL);
    SDL_SetWindowMouseGrab(g_win, want ? SDL_TRUE : SDL_FALSE);
    if (want) fprintf(stderr, "[screen] mouse held in the picture: %dx%d at %d,%d\n", r.w, r.h, r.x, r.y);
    else fprintf(stderr, "[screen] mouse let go\n");
    held = want, at = r;
#endif
}

#ifdef __EMSCRIPTEN__
#include <emscripten/threading.h>
#endif
int g_input_trace;                                         /* a host's --inputtrace */
/* Touch, for a game made for a mouse (a phone). SDL's own touch-to-mouse is
 * off (hle_screen_open); here one finger is the left button, two the right:
 *  - one finger: the cursor goes to it; nothing is pressed while it stays
 *    still. A move past TOUCH_SLOP presses the left button and drags (a
 *    selection box) until the lift; a lift before any move is a left click;
 *  - a second finger while the first still waits: no left press at all, and
 *    when the last finger lifts, a right click where the first was;
 *  - one finger held still TOUCH_STICK_MS: an analog stick comes up under it
 *    and the camera follows the finger's pull (the arrow keys, pulsed by how
 *    far: farther is faster) until the lift. A map scrolls at its edges for
 *    a mouse, which a finger can hardly reach. */
#define TOUCH_SLOP 0.035f                                /* of the window's size */
#define TOUCH_STICK_MS 1000                              /* a still finger this long: the stick (4 s was too long to hold still) */
#define STICK_RADIUS 0.36f                               /* of the window's height: full speed there */
#define STICK_DEAD 0.18f                                 /* of the radius: no scroll inside */
#define STICK_FULL 0.6f                                  /* of the radius: from here the keys are held (full speed) */
int (*hle_touch_stick_hook)(void);                         /* the host: may the stick come up now (a match, not a menu)? */
static int g_touch_log = 1;                              /* [touch] lines (few: one a touch) */
static SDL_FingerID g_t_first;
static int g_t_fingers, g_t_state;                       /* 0 none, 1 waiting, 2 left held, 3 right (two fingers), 4 stick */
static uint32_t g_t_since;
static float g_t_x0, g_t_y0, g_t_x, g_t_y;
static int g_t_gx, g_t_gy;
static uint32_t g_t_up_due;                              /* a tap's left button goes up then (0: none) */
#define TOUCH_TAP_HOLD_MS 80                             /* a menu polls its buttons by frame: down and up at once was missed */
static int g_stick_keys;                                 /* arrow keys held: 1 left, 2 up, 4 right, 8 down */
/* The ring menu: two fingers held still TOUCH_STICK_MS bring up, between
 * them, a ring the stick's size with the host's commands around it
 * (hle_radial_items_hook: a label and its key, as text). It stays up after
 * the lift; a tap on a command runs it (hle_radial_pick_hook: the host sends
 * its keys), a tap anywhere else only closes it. */
#define RADIAL_MAX 16
const hle_ring_item_t *(*hle_radial_items_hook)(int *n);
void (*hle_radial_pick_hook)(int i);
static int g_rad_open, g_rad_arming, g_rad_n, g_rad_hit = -1;
static const hle_ring_item_t *g_rad_items;

/* The hot key list (hle_screen_hotkeys): a panel over the picture, a title
 * and a close box on top, then one row a key (key, command, description),
 * dragged to scroll. A tap on a row runs it (the host's pick: its key) and
 * closes the list; a row that is not enabled (a key with a mouse click, a
 * key held) only shows. A tap on the close box or outside closes it. */
static const hle_hotkey_item_t *g_hk_items;
static int g_hk_n, g_hk_open, g_hk_down, g_hk_moved, g_hk_row = -1;
static void (*g_hk_pick)(int i);
static float g_hk_scroll, g_hk_y0, g_hk_scroll0, g_hk_x0;   /* scroll in pixels; the touch-down */

static float g_rad_fx, g_rad_fy;                         /* its centre, a share of the window */
static float g_t2_x, g_t2_y;                             /* the second finger's touch-down */
static SDL_FingerID g_t2_id;
static float g_t2_cx, g_t2_cy;                           /* ... and where it is now */
/* Two fingers that move (state 6): pinch zooms the picture about their
 * middle, both moving together drag it; the point under the fingers stays
 * under them. Lifting ends it with no right click (state 7 until all are up). */
static float g_zg_z0, g_zg_d0, g_zg_mx0, g_zg_my0;
static SDL_Rect g_zg_v0;
static void zoom_begin(void) {
    int w = 1, h = 1;
    if (g_ren) SDL_GetRendererOutputSize(g_ren, &w, &h);
    float dx = (g_t2_x - g_t_x0) * w, dy = (g_t2_y - g_t_y0) * h;
    g_zg_d0 = SDL_sqrtf(dx * dx + dy * dy);
    if (g_zg_d0 < 20.0f) g_zg_d0 = 20.0f;
    g_zg_mx0 = (g_t_x0 + g_t2_x) / 2 * w, g_zg_my0 = (g_t_y0 + g_t2_y) / 2 * h;
    g_zg_z0 = g_zoom, g_zg_v0 = g_dst;
    if (g_touch_log) fprintf(stderr, "[touch] zoom from %.2f\n", g_zoom);
}
static void zoom_update(void) {
    int w = 1, h = 1;
    if (g_ren) SDL_GetRendererOutputSize(g_ren, &w, &h);
    if (g_fit.w <= 0 || g_fit.h <= 0 || g_zg_v0.w <= 0 || g_zg_v0.h <= 0) return;
    float dx = (g_t2_cx - g_t_x) * w, dy = (g_t2_cy - g_t_y) * h, d = SDL_sqrtf(dx * dx + dy * dy);
    float mx = (g_t_x + g_t2_cx) / 2 * w, my = (g_t_y + g_t2_cy) / 2 * h;
    float z = g_zg_z0 * d / g_zg_d0;
    if (z < 1.03f) z = 1.0f;                             /* back to the original size */
    if (z > ZOOM_MAX) z = ZOOM_MAX;
    float ux = (g_zg_mx0 - g_zg_v0.x) / g_zg_v0.w, uy = (g_zg_my0 - g_zg_v0.y) / g_zg_v0.h;   /* the picture point under the start */
    float zw = g_fit.w * z, zh = g_fit.h * z, vx = mx - ux * zw, vy = my - uy * zh;
    if (vx > g_fit.x) vx = (float)g_fit.x;               /* the picture always covers the fit */
    if (vx < g_fit.x + g_fit.w - zw) vx = g_fit.x + g_fit.w - zw;
    if (vy > g_fit.y) vy = (float)g_fit.y;
    if (vy < g_fit.y + g_fit.h - zh) vy = g_fit.y + g_fit.h - zh;
    g_zoom = z;
    g_zpx = zw > g_fit.w + 0.5f ? (g_fit.x - vx) / (zw - g_fit.w) : 0.5f;
    g_zpy = zh > g_fit.h + 0.5f ? (g_fit.y - vy) / (zh - g_fit.h) : 0.5f;
    g_shown_seq = ~0u;
}
void hle_screen_zoom_reset(void) { g_zoom = 1.0f, g_zpx = g_zpy = 0.5f; g_shown_seq = ~0u; }
static uint32_t g_t2_since;
static int g_t_moved;                                    /* a finger left TOUCH_SLOP (two fingers) */
#define RADIAL_SIZE 1.0f                                 /* of the stick's radius */
static uint32_t g_stick_phase;
static void touch_to_game(float fx, float fy, int *gx, int *gy) {
    int ww = 0, wh = 0;
    SDL_GetWindowSize(g_win, &ww, &wh);
    to_game((int)(fx * ww), (int)(fy * wh), gx, gy);
}
static void touch_press_left(void) {
    hle_input_mouse(WM_LBUTTONDOWN, g_t_gx, g_t_gy);
    g_t_state = 2;
}
static void stick_keys(int want) {
    static const struct { int bit; uint32_t vk, scan; } k[4] = {
        { 1, 0x25, 75 }, { 2, 0x26, 72 }, { 4, 0x27, 77 }, { 8, 0x28, 80 } };   /* scan codes as scan_of gives them */
    for (int i = 0; i < 4; i++)
        if ((want ^ g_stick_keys) & k[i].bit) hle_input_key((want & k[i].bit) != 0, k[i].vk, k[i].scan);
    g_stick_keys = want;
}
/* the stick's pull, -1..1 each way, as a share of its radius (window frame: x scaled to y) */
static void stick_pull(float *px, float *py) {
    int ww = 1, wh = 1;
    SDL_GetWindowSize(g_win, &ww, &wh);
    float dx = (g_t_x - g_t_x0) * (float)ww / (float)(wh > 0 ? wh : 1), dy = g_t_y - g_t_y0;
    float d = SDL_sqrtf(dx * dx + dy * dy), r = STICK_RADIUS;
    if (d > r) dx *= r / d, dy *= r / d;
    *px = dx / r, *py = dy / r;
}
/* the ring's geometry in renderer pixels: centre, radius, item ring, item radius */
static void radial_geom(int *cx, int *cy, int *r, int *rr, int *ir) {
    int w = 1, h = 1;
    if (g_ren) SDL_GetRendererOutputSize(g_ren, &w, &h);
    int R = (int)(STICK_RADIUS * RADIAL_SIZE * h);
    int x = (int)(g_rad_fx * w), y = (int)(g_rad_fy * h);
    if (x < R) x = R;                                    /* the whole ring on the screen */
    if (x > w - R) x = w - R;
    if (y < R) y = R;
    if (y > h - R) y = h - R;
    int ring = R * 60 / 100, it = R * 36 / 100;
    if (g_rad_n > 0) {                                   /* items no wider than their share of the circle */
        int fit = (int)(3.14159f * ring / g_rad_n * 0.9f);
        if (fit < it) it = fit;
    }
    *cx = x, *cy = y, *r = R, *rr = ring, *ir = it;
}
static void radial_item_at(int i, int *x, int *y) {
    int cx, cy, r, rr, ir;
    radial_geom(&cx, &cy, &r, &rr, &ir);
    float a = -1.5707963f + 6.2831853f * (float)i / (float)(g_rad_n > 0 ? g_rad_n : 1);   /* from the top, clockwise */
    *x = cx + (int)(rr * SDL_cosf(a)), *y = cy + (int)(rr * SDL_sinf(a));
}
/* which item a touch (a share of the window) is on: its index, -1 none */
static int radial_hit(float fx, float fy) {
    int w = 1, h = 1, cx, cy, r, rr, ir;
    if (g_ren) SDL_GetRendererOutputSize(g_ren, &w, &h);
    radial_geom(&cx, &cy, &r, &rr, &ir);
    int px = (int)(fx * w), py = (int)(fy * h), best = -1;
    long long bd = 0, lim = (long long)(ir * 13 / 10) * (ir * 13 / 10);   /* a little past the drawn circle */
    for (int i = 0; i < g_rad_n; i++) {
        int x, y;
        radial_item_at(i, &x, &y);
        long long d = (long long)(px - x) * (px - x) + (long long)(py - y) * (py - y);
        if (d <= lim && (best < 0 || d < bd)) best = i, bd = d;
    }
    return best;
}
static void radial_open(float fx, float fy) {
    int n = 0;
    const hle_ring_item_t *it = hle_radial_items_hook ? hle_radial_items_hook(&n) : NULL;
    if (!it || n <= 0) { if (g_touch_log) fprintf(stderr, "[touch] ring: no commands\n"); return; }
    if (n > RADIAL_MAX) n = RADIAL_MAX;
    g_rad_items = it, g_rad_n = n, g_rad_open = 1, g_rad_arming = 1, g_rad_hit = -1, g_rad_fx = fx, g_rad_fy = fy;
    if (g_touch_log) fprintf(stderr, "[touch] ring on: %d commands\n", n);
    g_shown_seq = ~0u;
}
static void radial_close(void) { g_rad_open = 0, g_rad_hit = -1; g_shown_seq = ~0u; }
static void radial_click(int i) {
    if (g_touch_log) fprintf(stderr, "[touch] ring: %s (%s)\n", g_rad_items[i].label ? g_rad_items[i].label : "", g_rad_items[i].key ? g_rad_items[i].key : "");
    if (hle_radial_pick_hook) hle_radial_pick_hook(i);
}
/* the list's panel, title bar and close box, rows' height, in renderer pixels */
static void hk_geom(SDL_Rect *panel, SDL_Rect *close, int *title_h, int *row_h) {
    int w = 1, h = 1;
    if (g_ren) SDL_GetRendererOutputSize(g_ren, &w, &h);
    int pw = w * 86 / 100, ph = h * 90 / 100;
    *panel = (SDL_Rect){ (w - pw) / 2, (h - ph) / 2, pw, ph };
    *title_h = h / 9, *row_h = h / 8;
    int cs = *title_h * 80 / 100;
    *close = (SDL_Rect){ panel->x + pw - cs - *title_h / 10, panel->y + (*title_h - cs) / 2, cs, cs };
}
static float hk_max_scroll(void) {
    SDL_Rect pa, cl; int th, rh;
    hk_geom(&pa, &cl, &th, &rh);
    float m = (float)(g_hk_n * rh - (pa.h - th));
    return m > 0 ? m : 0;
}
/* the row under a point (renderer pixels): its index, -1 none */
static int hk_row_at(int x, int y) {
    SDL_Rect pa, cl; int th, rh;
    hk_geom(&pa, &cl, &th, &rh);
    if (x < pa.x || x >= pa.x + pa.w || y < pa.y + th || y >= pa.y + pa.h) return -1;
    int i = (int)((y - pa.y - th + g_hk_scroll) / rh);
    return i >= 0 && i < g_hk_n ? i : -1;
}
void hle_screen_hotkeys(const hle_hotkey_item_t *items, int n, void (*pick)(int i)) {
    if (!items || n <= 0) return;
    g_hk_items = items, g_hk_n = n, g_hk_pick = pick, g_hk_open = 1, g_hk_down = 0, g_hk_row = -1, g_hk_scroll = 0;
    if (g_touch_log) fprintf(stderr, "[touch] hot keys: %d rows\n", n);
    g_shown_seq = ~0u;
}
static void hk_close(void) { g_hk_open = 0, g_hk_down = 0, g_hk_row = -1; g_shown_seq = ~0u; }
/* a touch on the list: drag scrolls, a tap runs a row or closes */
static void hk_touch(const SDL_Event *e) {
    const SDL_TouchFingerEvent *f = &e->tfinger;
    int w = 1, h = 1;
    if (g_ren) SDL_GetRendererOutputSize(g_ren, &w, &h);
    int x = (int)(f->x * w), y = (int)(f->y * h);
    g_t_fingers = e->type == SDL_FINGERMOTION ? g_t_fingers : SDL_GetNumTouchFingers(f->touchId);
    if (e->type == SDL_FINGERDOWN) {
        if (g_hk_down) return;                           /* one finger drives it */
        g_t_first = f->fingerId, g_hk_down = 1, g_hk_moved = 0;
        g_hk_x0 = f->x, g_hk_y0 = f->y, g_hk_scroll0 = g_hk_scroll;
        g_hk_row = hk_row_at(x, y);
        g_shown_seq = ~0u;
        return;
    }
    if (f->fingerId != g_t_first || !g_hk_down) return;
    if (e->type == SDL_FINGERMOTION) {
        if (SDL_fabs(f->y - g_hk_y0) > TOUCH_SLOP || SDL_fabs(f->x - g_hk_x0) > TOUCH_SLOP) g_hk_moved = 1, g_hk_row = -1;
        if (g_hk_moved) {
            float sc = g_hk_scroll0 - (f->y - g_hk_y0) * h, m = hk_max_scroll();
            g_hk_scroll = sc < 0 ? 0 : sc > m ? m : sc;
        }
        g_shown_seq = ~0u;
        return;
    }
    if (e->type == SDL_FINGERUP) {
        g_hk_down = 0;
        if (g_hk_moved) { g_hk_row = -1; g_shown_seq = ~0u; return; }
        SDL_Rect pa, cl; int th, rh;
        hk_geom(&pa, &cl, &th, &rh);
        int in_close = x >= cl.x - cl.w / 2 && x < cl.x + cl.w + cl.w / 2 && y >= cl.y - cl.h / 2 && y < cl.y + cl.h + cl.h / 2;
        int out = x < pa.x || x >= pa.x + pa.w || y < pa.y || y >= pa.y + pa.h;
        int row = hk_row_at(x, y), was = g_hk_row;
        if (g_touch_log) fprintf(stderr, "[touch] hot keys tap at %d,%d: row %d%s%s\n", x, y, row, in_close ? " (close)" : "", out ? " (outside)" : "");
        if (in_close || out) { hk_close(); return; }
        if (row >= 0 && row == was && g_hk_items[row].enabled) {
            hk_close();
            if (g_touch_log) fprintf(stderr, "[touch] hot key: %s (%s)\n", g_hk_items[row].command, g_hk_items[row].key);
            if (g_hk_pick) g_hk_pick(row);
            return;
        }
        g_hk_row = -1;
        g_shown_seq = ~0u;
    }
}
static void touch_event(const SDL_Event *e) {
    const SDL_TouchFingerEvent *rf = &e->tfinger;
    if (g_hk_open) { hk_touch(e); g_t_state = 0; return; }
    if (g_rad_open) {                                    /* the ring is up: a tap picks or closes, nothing reaches the game */
        if (g_rad_arming) {                              /* the fingers that opened it: wait for them all to lift */
            if (e->type != SDL_FINGERMOTION) g_t_fingers = SDL_GetNumTouchFingers(rf->touchId);
            if (e->type == SDL_FINGERUP && g_t_fingers == 0) {
                g_rad_arming = 0;
                if (g_touch_log) fprintf(stderr, "[touch] ring: fingers up, waiting for a tap\n");
            }
            g_t_state = 0;
            return;
        }
        if (e->type == SDL_FINGERDOWN) {
            g_t_fingers = SDL_GetNumTouchFingers(rf->touchId);
            if (g_t_fingers <= 1) g_t_first = rf->fingerId, g_rad_hit = radial_hit(rf->x, rf->y);
            else g_rad_hit = -2;                         /* more fingers: not a tap */
        } else if (e->type == SDL_FINGERUP) {
            g_t_fingers = SDL_GetNumTouchFingers(rf->touchId);
            if (rf->fingerId == g_t_first && g_rad_hit != -2) {
                int at = radial_hit(rf->x, rf->y), hit = g_rad_hit;
                if (g_touch_log) fprintf(stderr, "[touch] ring tap at %.3f,%.3f: item %d (down on %d)\n", rf->x, rf->y, at, hit);
                radial_close();
                if (at >= 0 && (at == hit || hit == -1)) radial_click(at);   /* down just outside a circle, up on it: still that one */
            } else if (g_t_fingers == 0) radial_close();
        }
        g_t_state = 0;
        return;
    }
    const SDL_TouchFingerEvent *f = &e->tfinger;
    if (e->type == SDL_FINGERDOWN) {
        if (g_touch_log) fprintf(stderr, "[touch] down %d at %.3f,%.3f (state %d)\n", g_t_fingers + 1, f->x, f->y, g_t_state);
        g_t_fingers = SDL_GetNumTouchFingers(f->touchId);   /* SDL's count: a missed lift once left ours stuck */
        if (g_t_up_due) hle_input_mouse(WM_LBUTTONUP, g_t_gx, g_t_gy), g_t_up_due = 0;
        if (g_t_fingers <= 1 || g_t_state == 0) {
            g_t_first = f->fingerId, g_t_state = 1, g_t_since = SDL_GetTicks();
            g_t_x0 = g_t_x = f->x, g_t_y0 = g_t_y = f->y;
            touch_to_game(f->x, f->y, &g_t_gx, &g_t_gy);
            hle_input_mouse(WM_MOUSEMOVE, g_t_gx, g_t_gy);
        } else if (g_t_state == 1) {
            g_t_state = 3;                               /* two fingers: a right click when they lift */
            g_t2_x = g_t2_cx = f->x, g_t2_y = g_t2_cy = f->y, g_t2_since = SDL_GetTicks(), g_t_moved = 0;
            g_t2_id = f->fingerId;
        }
        return;
    }
    if (e->type == SDL_FINGERMOTION) {
        if (g_t_state == 3 || g_t_state == 6) {          /* two fingers: where each is now */
            if (f->fingerId == g_t_first) g_t_x = f->x, g_t_y = f->y;
            else if (f->fingerId == g_t2_id) g_t2_cx = f->x, g_t2_cy = f->y;
        }
        if (g_t_state == 3 && (f->fingerId == g_t_first ? SDL_fabs(f->x - g_t_x0) > TOUCH_SLOP || SDL_fabs(f->y - g_t_y0) > TOUCH_SLOP
                                                       : SDL_fabs(f->x - g_t2_x) > TOUCH_SLOP || SDL_fabs(f->y - g_t2_y) > TOUCH_SLOP))
            g_t_moved = 1;                               /* two fingers that move: no ring */
        if (g_t_state == 3 && g_t_moved && g_t_fingers >= 2) g_t_state = 6, zoom_begin();
        if (g_t_state == 6) { zoom_update(); return; }
        if (f->fingerId != g_t_first || g_t_state == 3 || g_t_state == 5 || g_t_state == 7 || g_t_state == 0) return;
        g_t_x = f->x, g_t_y = f->y;
        if (g_t_state == 4) return;                      /* the stick: touch_poll scrolls */
        touch_to_game(f->x, f->y, &g_t_gx, &g_t_gy);
        if (g_t_state == 1 && (SDL_fabs(f->x - g_t_x0) > TOUCH_SLOP || SDL_fabs(f->y - g_t_y0) > TOUCH_SLOP)) {
            if (g_touch_log) fprintf(stderr, "[touch] drag after %u ms\n", SDL_GetTicks() - g_t_since);
            touch_press_left();
        }
        hle_input_mouse(WM_MOUSEMOVE, g_t_gx, g_t_gy);
        return;
    }
    if (e->type == SDL_FINGERUP) {
        if (g_touch_log) fprintf(stderr, "[touch] up (state %d, %u ms)\n", g_t_state, SDL_GetTicks() - g_t_since);
        g_t_fingers = SDL_GetNumTouchFingers(f->touchId);
        if (f->fingerId == g_t_first && g_t_state != 3) {
            if (g_t_state == 1) {                        /* a tap: down now, up a little later (touch_poll) */
                touch_press_left();
                g_t_up_due = SDL_GetTicks() + TOUCH_TAP_HOLD_MS;
                if (!g_t_up_due) g_t_up_due = 1;
            } else if (g_t_state == 2) hle_input_mouse(WM_LBUTTONUP, g_t_gx, g_t_gy);
            if (g_t_state == 4) stick_keys(0), g_shown_seq = ~0u;
            g_t_state = 0;
        }
        if (g_t_fingers == 0 && g_t_state == 5) g_t_state = 0;   /* the hold opened the ring: no right click */
        if (g_t_state == 6 && g_t_fingers < 2) {         /* the zoom ends: no right click */
            if (g_touch_log) fprintf(stderr, "[touch] zoom %.2f\n", g_zoom);
            g_t_state = g_t_fingers ? 7 : 0;
        }
        if (g_t_fingers == 0 && g_t_state == 3) {
            hle_input_mouse(WM_RBUTTONDOWN, g_t_gx, g_t_gy);
            hle_input_mouse(WM_RBUTTONUP, g_t_gx, g_t_gy);
            g_t_state = 0;
        }
        if (g_t_fingers == 0) {
            if (g_t_state == 4) stick_keys(0);
            g_t_state = 0;
        }
    }
}
/* a still finger long enough: the stick; the stick held: its arrow keys, in
 * eight ways, pulsed below STICK_FULL (see below) */
static void touch_poll(void) {
    uint32_t now = SDL_GetTicks();
    if (g_t_up_due && (int32_t)(now - g_t_up_due) >= 0) {
        hle_input_mouse(WM_LBUTTONUP, g_t_gx, g_t_gy);
        g_t_up_due = 0;
    }
    if (g_t_state == 1 && now - g_t_since >= TOUCH_STICK_MS && (!hle_touch_stick_hook || hle_touch_stick_hook())) {
        g_t_state = 4, g_stick_phase = now;
        if (g_touch_log) fprintf(stderr, "[touch] stick on\n");
        g_shown_seq = ~0u;                               /* draw it now */
    }
    if (g_t_state == 3 && !g_t_moved && g_t_fingers >= 2 && now - g_t2_since >= TOUCH_STICK_MS
        && (!hle_touch_stick_hook || hle_touch_stick_hook())) {
        g_t_state = 5;                                   /* the fingers' lift ends it, with no right click */
        radial_open((g_t_x0 + g_t2_x) / 2, (g_t_y0 + g_t2_y) / 2);
    }
    if (g_rad_open || g_hk_open) g_shown_seq = ~0u;      /* a still picture is drawn again, with the ring or the list */
    if (g_t_state != 4) return;
    g_shown_seq = ~0u;                                   /* a still picture (a menu) is drawn again, with the stick */
    float px, py;
    stick_pull(&px, &py);
    /* Eight ways, one duty for both keys: a pull near a diagonal holds both
     * arrows together (each axis pulsed apart zig-zagged, one key on while
     * the other was off). From STICK_FULL of the radius the keys are held
     * (the game's own scroll rate, smooth); below it they pulse in a 40 ms
     * cycle (100 ms stopped and went, five game frames at a time). */
    float mag = SDL_sqrtf(px * px + py * py);
    int want = 0;
    if (mag > STICK_DEAD) {
        static const int dirs[8] = { 4, 4 | 8, 8, 1 | 8, 1, 1 | 2, 2, 4 | 2 };   /* from right, clockwise (y down) */
        int sector = (int)SDL_floorf(SDL_atan2f(py, px) / 0.78539816f + 0.5f) & 7;
        float duty = (mag - STICK_DEAD) / (STICK_FULL - STICK_DEAD);
        float t = (float)((now - g_stick_phase) % 40) / 40.0f;
        if (duty >= 1.0f || t < duty) want = dirs[sector];
    }
    stick_keys(want);
}
/* the stick over the picture, while it is held: a ring and a knob */
static void fill_circle(int cx, int cy, int r) {
    for (int y = -r; y <= r; y++) {
        int x = (int)SDL_sqrt((double)(r * r - y * y));
        SDL_RenderDrawLine(g_ren, cx - x, cy + y, cx + x, cy + y);
    }
}
static void draw_ring(int cx, int cy, int r, int th) {
    int steps = r * 6 > 90 ? r * 6 : 90;
    for (int i = 0; i < steps; i++) {
        double a = i * 6.2831853 / steps;
        SDL_Rect d = { cx + (int)(r * SDL_cos(a)) - th / 2, cy + (int)(r * SDL_sin(a)) - th / 2, th, th };
        SDL_RenderFillRect(g_ren, &d);
    }
}
/* Text for the ring and the hot key list: a built-in 5x7 font, upper case,
 * drawn as squares of px pixels (no font file: a phone's system fonts are not
 * readable by the app; the same glyphs as the UWP launcher's). */
static const struct { char c; uint8_t row[7]; } k_bm[] = {
    {'A',{14,17,17,31,17,17,17}}, {'B',{30,17,17,30,17,17,30}}, {'C',{14,17,16,16,16,17,14}},
    {'D',{30,17,17,17,17,17,30}}, {'E',{31,16,16,30,16,16,31}}, {'F',{31,16,16,30,16,16,16}},
    {'G',{14,17,16,23,17,17,15}}, {'H',{17,17,17,31,17,17,17}}, {'I',{14,4,4,4,4,4,14}},
    {'J',{7,2,2,2,2,18,12}},      {'K',{17,18,20,24,20,18,17}}, {'L',{16,16,16,16,16,16,31}},
    {'M',{17,27,21,21,17,17,17}}, {'N',{17,17,25,21,19,17,17}}, {'O',{14,17,17,17,17,17,14}},
    {'P',{30,17,17,30,16,16,16}}, {'Q',{14,17,17,17,21,18,13}}, {'R',{30,17,17,30,20,18,17}},
    {'S',{15,16,16,14,1,1,30}},   {'T',{31,4,4,4,4,4,4}},       {'U',{17,17,17,17,17,17,14}},
    {'V',{17,17,17,17,17,10,4}},  {'W',{17,17,17,21,21,21,10}}, {'X',{17,17,10,4,10,17,17}},
    {'Y',{17,17,10,4,4,4,4}},     {'Z',{31,1,2,4,8,16,31}},
    {'0',{14,17,19,21,25,17,14}}, {'1',{4,12,4,4,4,4,14}},      {'2',{14,17,1,2,4,8,31}},
    {'3',{31,2,4,2,1,17,14}},     {'4',{2,6,10,18,31,2,2}},     {'5',{31,16,30,1,1,17,14}},
    {'6',{6,8,16,30,17,17,14}},   {'7',{31,1,2,4,8,8,8}},       {'8',{14,17,17,14,17,17,14}},
    {'9',{14,17,17,15,1,2,12}},   {'\'',{4,4,8,0,0,0,0}},      {':',{0,12,12,0,12,12,0}},
    {'-',{0,0,0,31,0,0,0}},       {'/',{1,1,2,4,8,16,16}},      {'+',{0,4,4,31,4,4,0}},
    {'(',{2,4,8,8,8,4,2}},        {')',{8,4,2,2,2,4,8}},        {'.',{0,0,0,0,0,12,12}},
    {',',{0,0,0,0,12,4,8}},       {'?',{14,17,1,2,4,0,4}},
};
static int bm_w(int px, const char *s) { int n = s ? (int)strlen(s) : 0; return n ? n * 6 * px - px : 0; }
/* s from x (its top at y), in the colour set; no wider than maxw (cut at a whole letter) */
static void bm_text(int x, int y, int px, const char *s, int maxw) {
    if (!s) return;
    for (int x0 = x; *s && x + 5 * px <= x0 + maxw; s++, x += 6 * px) {
        char c = (char)SDL_toupper((unsigned char)*s);
        for (size_t g = 0; g < sizeof k_bm / sizeof k_bm[0]; g++) {
            if (k_bm[g].c != c) continue;
            for (int row = 0; row < 7; row++)
                for (int col = 0; col < 5; col++)
                    if (k_bm[g].row[row] >> (4 - col) & 1) {
                        SDL_Rect d = { x + col * px, y + row * px, px, px };
                        SDL_RenderFillRect(g_ren, &d);
                    }
            break;
        }
    }
}
/* s centred on cx, words wrapped to wrap pixels, from top; returns the height drawn (or that would be, draw 0) */
static int bm_wrapped(int cx, int top, int px, const char *s, int wrap, int draw) {
    if (!s || !*s) return 0;
    int per = (wrap + px) / (6 * px), lines = 0;         /* letters a line */
    if (per < 1) per = 1;
    const char *p = s;
    while (*p) {
        while (*p == ' ') p++;
        if (!*p) break;
        int n = (int)strlen(p);
        if (n > per) {                                   /* break at the last space that fits */
            int k = per;
            while (k > 0 && p[k] != ' ') k--;
            n = k > 0 ? k : per;
        }
        if (draw) {
            char line[128];
            int m = n < (int)sizeof line - 1 ? n : (int)sizeof line - 1;
            memcpy(line, p, (size_t)m), line[m] = 0;
            bm_text(cx - bm_w(px, line) / 2, top + lines * 9 * px, px, line, wrap + px);
        }
        p += n, lines++;
    }
    return lines ? lines * 9 * px - 2 * px : 0;
}
/* a command in its circle: its key large (a word, as "Hot Keys", smaller), its name small under it */
static void radial_label(const char *label, const char *key, int x, int y, int ir) {
    int wrap = ir * 17 / 10;
    int kp = ir * 45 / 100 / 7;                          /* the key's letters: near half the circle high */
    if (key && bm_w(kp, key) > wrap) kp = wrap / (6 * (int)strlen(key));
    if (kp < 2) kp = 2;
    int lp = ir * 15 / 100 / 7 > 1 ? ir * 15 / 100 / 7 : 1;
    int h1 = key && *key ? 7 * kp : 0, h2 = bm_wrapped(x, 0, lp, label, wrap, 0), gap = h2 ? 3 * lp : 0;
    int top = y - (h1 + gap + h2) / 2;
    SDL_SetRenderDrawColor(g_ren, 255, 220, 80, 255);
    if (h1) bm_text(x - bm_w(kp, key) / 2, top, kp, key, wrap + kp);
    SDL_SetRenderDrawColor(g_ren, 230, 230, 230, 255);
    bm_wrapped(x, top + h1 + gap, lp, label, wrap, 1);
}
static void radial_draw(void) {
    if (!g_rad_open || !g_ren) return;
    int cx, cy, r, rr, ir;
    radial_geom(&cx, &cy, &r, &rr, &ir);
    int th = r / 30 > 2 ? r / 30 : 2;
    SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 140);
    fill_circle(cx, cy, r);
    SDL_SetRenderDrawColor(g_ren, 255, 255, 255, 200);
    draw_ring(cx, cy, r, th);
    for (int i = 0; i < g_rad_n; i++) {
        int x, y;
        radial_item_at(i, &x, &y);
        SDL_SetRenderDrawColor(g_ren, 20, 30, 60, 220);
        fill_circle(x, y, ir);
        radial_label(g_rad_items[i].label, g_rad_items[i].key, x, y, ir);
        if (g_rad_hit == i) SDL_SetRenderDrawColor(g_ren, 255, 220, 80, 240);
        else SDL_SetRenderDrawColor(g_ren, 90, 160, 255, 220);
        draw_ring(x, y, ir, th);
    }
    SDL_SetRenderDrawColor(g_ren, 255, 255, 255, 160);   /* the middle: an x (a tap outside the buttons closes) */
    {
        int c = r / 8;
        for (int t = -th / 2; t <= th / 2; t++) {
            SDL_RenderDrawLine(g_ren, cx - c + t, cy - c, cx + c + t, cy + c);
            SDL_RenderDrawLine(g_ren, cx + c + t, cy - c, cx - c + t, cy + c);
        }
    }
    SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_NONE);
}
/* a cell of the list: s from x, centred on y, cut at maxw */
static void hk_text(int px, const char *s, int x, int y, int maxw, SDL_Color c) {
    if (!s || !*s) return;
    SDL_SetRenderDrawColor(g_ren, c.r, c.g, c.b, c.a);
    bm_text(x, y - 7 * px / 2, px, s, maxw);
}
static void hk_draw(void) {
    if (!g_hk_open || !g_ren) return;
    SDL_Rect pa, cl; int th, rh;
    hk_geom(&pa, &cl, &th, &rh);
    int px = rh * 36 / 100 / 7 > 2 ? rh * 36 / 100 / 7 : 2, spx = px * 2 / 3 > 1 ? px * 2 / 3 : 1;   /* letters: key and command; description */
    SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(g_ren, 8, 12, 24, 230);
    SDL_RenderFillRect(g_ren, &pa);
    SDL_Rect bar = { pa.x, pa.y, pa.w, th };
    SDL_SetRenderDrawColor(g_ren, 30, 60, 120, 240);
    SDL_RenderFillRect(g_ren, &bar);
    SDL_SetRenderDrawColor(g_ren, 90, 160, 255, 220);
    SDL_RenderDrawRect(g_ren, &pa);
    int pad = th / 4;
    hk_text(px, "Hot Keys", pa.x + pad, pa.y + th / 2, pa.w / 2, (SDL_Color){ 255, 255, 255, 255 });
    SDL_SetRenderDrawColor(g_ren, 200, 60, 60, 240);     /* the close box: a red square with an x */
    SDL_RenderFillRect(g_ren, &cl);
    SDL_SetRenderDrawColor(g_ren, 255, 255, 255, 255);
    {
        int m = cl.w / 4, t = cl.w / 12 > 1 ? cl.w / 12 : 1;
        for (int k = -t; k <= t; k++) {
            SDL_RenderDrawLine(g_ren, cl.x + m + k, cl.y + m, cl.x + cl.w - m + k, cl.y + cl.h - m);
            SDL_RenderDrawLine(g_ren, cl.x + cl.w - m + k, cl.y + m, cl.x + m + k, cl.y + cl.h - m);
        }
    }
    SDL_Rect body = { pa.x, pa.y + th, pa.w, pa.h - th };
    SDL_RenderSetClipRect(g_ren, &body);
    int ck = pa.w * 18 / 100, cc = pa.w * 26 / 100;      /* columns: key, command, description */
    int first = (int)(g_hk_scroll / rh);
    for (int i = first; i < g_hk_n; i++) {
        int y = body.y + i * rh - (int)g_hk_scroll;
        if (y >= body.y + body.h) break;
        const hle_hotkey_item_t *it = &g_hk_items[i];
        if (i == g_hk_row && it->enabled) {
            SDL_Rect hl = { pa.x + 1, y, pa.w - 2, rh };
            SDL_SetRenderDrawColor(g_ren, 60, 110, 200, 200);
            SDL_RenderFillRect(g_ren, &hl);
        }
        SDL_SetRenderDrawColor(g_ren, 60, 80, 120, 160);
        SDL_RenderDrawLine(g_ren, pa.x + pad, y + rh - 1, pa.x + pa.w - pad, y + rh - 1);
        SDL_Color kc = it->enabled ? (SDL_Color){ 255, 220, 80, 255 } : (SDL_Color){ 150, 150, 150, 255 };
        SDL_Color tc = it->enabled ? (SDL_Color){ 255, 255, 255, 255 } : (SDL_Color){ 150, 150, 150, 255 };
        hk_text(px, it->key, pa.x + pad, y + rh / 2, ck - pad, kc);
        hk_text(px, it->command, pa.x + ck, y + rh / 2, cc - pad, tc);
        hk_text(spx, it->desc, pa.x + ck + cc, y + rh / 2, pa.w - ck - cc - pad, (SDL_Color){ 190, 190, 200, 255 });
    }
    SDL_RenderSetClipRect(g_ren, NULL);
    float m = hk_max_scroll();                           /* the scroll bar */
    if (m > 0) {
        int bh = body.h * body.h / (body.h + (int)m), by = body.y + (int)((body.h - bh) * (g_hk_scroll / m));
        SDL_Rect sb = { pa.x + pa.w - pad / 2 - 2, by, pad / 2 > 3 ? pad / 2 : 3, bh };
        SDL_SetRenderDrawColor(g_ren, 150, 180, 230, 160);
        SDL_RenderFillRect(g_ren, &sb);
    }
    SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_NONE);
}
static void touch_draw(void) {
    hk_draw();
    radial_draw();
    if (g_t_state != 4 || !g_ren) return;
    int w, h;
    SDL_GetRendererOutputSize(g_ren, &w, &h);
    float px, py;
    stick_pull(&px, &py);
    int cx = (int)(g_t_x0 * w), cy = (int)(g_t_y0 * h), r = (int)(STICK_RADIUS * h), kr = r / 3;
    int kx = cx + (int)(px * r), ky = cy + (int)(py * r);
    SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_BLEND);
    SDL_SetRenderDrawColor(g_ren, 0, 0, 0, 110);
    for (int y = -r; y <= r; y++) {                      /* the base, filled */
        int x = (int)SDL_sqrt((double)(r * r - y * y));
        SDL_RenderDrawLine(g_ren, cx - x, cy + y, cx + x, cy + y);
    }
    SDL_SetRenderDrawColor(g_ren, 255, 255, 255, 200);
    int th = r / 25 > 2 ? r / 25 : 2;                    /* its ring, a few pixels thick */
    for (int i = 0; i < 720; i++) {
        double a = i * 3.14159265 / 360.0;
        SDL_Rect d = { cx + (int)(r * SDL_cos(a)) - th / 2, cy + (int)(r * SDL_sin(a)) - th / 2, th, th };
        SDL_RenderFillRect(g_ren, &d);
    }
    SDL_SetRenderDrawColor(g_ren, 255, 255, 255, 170);
    for (int y = -kr; y <= kr; y++) {                    /* the knob */
        int x = (int)SDL_sqrt((double)(kr * kr - y * y));
        SDL_RenderDrawLine(g_ren, kx - x, ky + y, kx + x, ky + y);
    }
    SDL_SetRenderDrawBlendMode(g_ren, SDL_BLENDMODE_NONE);
}

void hle_screen_pump(void) {
#ifdef __EMSCRIPTEN__
    /* the browser's main thread queues this thread's input callbacks */
    emscripten_current_thread_process_queued_calls();
    web_menu_poll();
#endif
    SDL_Event e;
    if (g_win) update_cursor();
    while (g_win && SDL_PollEvent(&e)) {
#ifdef __EMSCRIPTEN__
        if (getenv("RA2_INPUT_TRACE") || g_input_trace) {
            if (e.type == SDL_KEYDOWN || e.type == SDL_KEYUP) fprintf(stderr, "[input] key %s scancode %d repeat %d\n", e.type == SDL_KEYDOWN ? "down" : "up", e.key.keysym.scancode, e.key.repeat);
            else if (e.type == SDL_MOUSEBUTTONDOWN || e.type == SDL_MOUSEBUTTONUP) fprintf(stderr, "[input] button %d %s at %d,%d\n", e.button.button, e.type == SDL_MOUSEBUTTONDOWN ? "down" : "up", e.button.x, e.button.y);
        }
#endif
        if (g_menu_depth && menu_event(&e)) continue;
        if (g_menu_eat_up && e.type == SDL_MOUSEBUTTONUP) { g_menu_eat_up = 0; continue; }
        if (e.type == SDL_FINGERDOWN || e.type == SDL_FINGERUP || e.type == SDL_FINGERMOTION) { touch_event(&e); continue; }
        if ((e.type == SDL_MOUSEBUTTONDOWN || e.type == SDL_MOUSEBUTTONUP) && e.button.which == SDL_TOUCH_MOUSEID) continue;
        if (e.type == SDL_MOUSEMOTION && e.motion.which == SDL_TOUCH_MOUSEID) continue;
        switch (e.type) {
        case SDL_QUIT: {
            uint32_t h = hle_first_hwnd();
            if (h) hle_post_message(h, WM_CLOSE, 0, 0); else hle_exit(0);
            break;
        }
        case SDL_KEYDOWN: case SDL_KEYUP: {
            SDL_Scancode sc = e.key.keysym.scancode;
            /* a key the host keeps for itself (a mod switch): the game never sees it */
            if (hle_screen_key_hook && hle_screen_key_hook((int)sc, e.type == SDL_KEYDOWN)) break;
            /* Ctrl+Cmd (Ctrl+Win, Ctrl+Super), as a virtual machine's: let the mouse go */
            if (e.type == SDL_KEYDOWN && (e.key.keysym.mod & KMOD_CTRL) && (e.key.keysym.mod & KMOD_GUI)
                && (sc == SDL_SCANCODE_LCTRL || sc == SDL_SCANCODE_RCTRL || sc == SDL_SCANCODE_LGUI || sc == SDL_SCANCODE_RGUI)) {
                if (!g_released) fprintf(stderr, "[screen] mouse let go (" RELEASE_KEYS "); click the window to hold it again\n");
                g_released = 1;
                break;
            }
            /* the presenter's keys, the Windows build's: they do not reach the game */
            if ((sc == SDL_SCANCODE_RETURN && (e.key.keysym.mod & KMOD_ALT)) || sc == SDL_SCANCODE_F11) {
                if (e.type == SDL_KEYDOWN) set_fullscreen(!g_fullscreen);
                break;
            }
            if (sc == SDL_SCANCODE_F12) {
                if (e.type == SDL_KEYDOWN) {
                    g_mode = (g_mode + 1) % 5;
                    fprintf(stderr, "[screen] scaling: %s\n", k_modes[g_mode]);
                    settings_changed();
                }
                break;
            }
            int vk = vk_of(e.key.keysym.scancode);
            if (vk) hle_input_key(e.type == SDL_KEYDOWN, (uint32_t)vk, scan_of(e.key.keysym.scancode));
            break;
        }
        case SDL_MOUSEMOTION: {
            if (g_rel) {                                 /* movement alone, in game pixels as the picture shows them */
                SDL_Rect pd = to_points(g_dst);
                int w = g_src_w, h = g_src_h, bpp;
                if (w <= 0 || h <= 0) hle_dd_mode(&w, &h, &bpp);
                if (pd.w <= 0 || pd.h <= 0) break;
                g_rel_fx += (double)e.motion.xrel * w / pd.w, g_rel_fy += (double)e.motion.yrel * h / pd.h;
                int dx = (int)g_rel_fx, dy = (int)g_rel_fy;
                g_rel_fx -= dx, g_rel_fy -= dy;
                if (dx || dy) hle_input_motion(dx, dy);
                break;
            }
            int gx, gy;
            to_game(e.motion.x, e.motion.y, &gx, &gy);
            hle_input_mouse(WM_MOUSEMOVE, gx, gy);
            break;
        }
        case SDL_MOUSEBUTTONDOWN: case SDL_MOUSEBUTTONUP: {
            if (g_released && g_confine && e.type == SDL_MOUSEBUTTONDOWN) {   /* the click that takes the mouse back */
                g_released = 0, g_regrab_up = 1;
                break;
            }
            if (g_regrab_up && e.type == SDL_MOUSEBUTTONUP) { g_regrab_up = 0; break; }
            int gx, gy, down = e.type == SDL_MOUSEBUTTONDOWN;
            if (g_rel) hle_input_position(&gx, &gy);       /* where the movement has put the cursor */
            else to_game(e.button.x, e.button.y, &gx, &gy);
            uint32_t m = e.button.button == SDL_BUTTON_RIGHT ? (down ? WM_RBUTTONDOWN : WM_RBUTTONUP)
                       : e.button.button == SDL_BUTTON_MIDDLE ? (down ? WM_MBUTTONDOWN : WM_MBUTTONUP)
                       : (down ? WM_LBUTTONDOWN : WM_LBUTTONUP);
            hle_input_mouse(m, gx, gy);
            break;
        }
        case SDL_MOUSEWHEEL: {                           /* a notch is 120, as Windows counts it */
            int y = e.wheel.direction == SDL_MOUSEWHEEL_FLIPPED ? -e.wheel.y : e.wheel.y;
            if (y) hle_input_wheel(y * 120);
            break;
        }
        case SDL_WINDOWEVENT:
            if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) g_menu_depth = 0;   /* placed for the old size */
            if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED || e.window.event == SDL_WINDOWEVENT_EXPOSED) g_shown_seq = ~0u;
            break;
        default: break;
        }
    }
    touch_poll();
    if (hle_screen_poll_hook) hle_screen_poll_hook();
    present(0);
    confine_update();
}

static void present_gdi(void) { present(0); }

/* The guest moved the cursor (SetCursorPos): move the real one with it. */
static void warp(int x, int y) {
    int w, h, bpp;
    hle_dd_mode(&w, &h, &bpp);
    if (!g_win || !g_ren || g_rel || g_dst.w <= 0 || w <= 0 || h <= 0) return;   /* relative: there is no pointer to move */
    SDL_Rect p = to_points((SDL_Rect){ g_dst.x + x * g_dst.w / w, g_dst.y + y * g_dst.h / h, 0, 0 });
    SDL_WarpMouseInWindow(g_win, p.x, p.y);
}

/* The game drew into the primary. On the window's thread, show it (present
 * keeps to 60 a second, except a flip, which is always shown), and keep the
 * window's own events moving: a load
 * that does not pump for seconds would otherwise look hung to the desktop
 * (macOS's spinning cursor). The events wait in SDL's queue for the pump. */
static void primary_changed(void) {
    static uint32_t last_events;
    if (!g_ren || !pthread_equal(pthread_self(), g_video_thread)) return;
    uint32_t now = hle_ticks_ms();
    int flip = hle_dd_flipping();
    if (!flip && now - g_last_present < present_ms()) return;
    if (now - last_events >= 100) {
        PT_START();
        SDL_PumpEvents(), last_events = now;
        PT(5);
        hle_present_n[5]++;
    }
    present(flip);
}

int hle_screen_open(const char *title, int fullscreen, int headless) {
    g_headless = headless, g_fullscreen = fullscreen;
    SDL_SetHint(SDL_HINT_TOUCH_MOUSE_EVENTS, "0");      /* touch is ours (touch_event): two fingers, a right click */
    hle_set_pump_hook(hle_screen_pump);
    hle_input_relative_hook = want_relative;
    hle_gdi_window_hook = present_gdi;                   /* a GDI game's blit shows, pumping or not (a loading bar) */
    if (headless) return 0;
    SDL_SetHint(SDL_HINT_NO_SIGNAL_HANDLERS, "1");      /* SIGINT and SIGTERM end the process: SDL would make them a close the game can refuse */
    SDL_SetHint(SDL_HINT_VIDEO_MAC_FULLSCREEN_SPACES, "1"); /* macOS: full screen in its own Space, as apps do */
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "[screen] SDL: %s\n", SDL_GetError()); return -1; }
    g_win = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 960,
                             SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI | (fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0)
                             | (hle_screen_gl == 2 ? SDL_WINDOW_VULKAN : hle_screen_gl ? SDL_WINDOW_OPENGL : 0));
    if (!g_win) { fprintf(stderr, "[screen] window: %s\n", SDL_GetError()); return -1; }
    if (hle_screen_gl) {                                 /* the host's GL layer draws; no SDL renderer */
        SDL_ShowCursor(SDL_DISABLE);
        hle_cursor_hook = warp;
        g_video_thread = pthread_self();
        fprintf(stderr, "[screen] %s, %s window (the host draws), fullscreen F11\n", SDL_GetCurrentVideoDriver(),
                hle_screen_gl == 2 ? "Vulkan" : "OpenGL");
        return 0;
    }
#ifdef __EMSCRIPTEN__
    /* In a browser the game runs on a worker thread and the canvas belongs
     * to the page's: SDL's GL renderer would need the WebGL context made
     * current on this thread, which its EGL path (proxied to the page's
     * thread) does not do. The software renderer draws into the window's
     * framebuffer, which SDL hands to the canvas from the page's thread. */
    g_ren = SDL_CreateRenderer(g_win, -1, SDL_RENDERER_SOFTWARE);
#else
    /* HLE_VSYNC=0: present without waiting for the display. The present runs
     * on the game's thread, so a vsync wait there is time the game logic
     * does not run (on a slow device it was ~40% of the main thread). */
    {
        const char *v = SDL_getenv("HLE_VSYNC");
        g_ren = SDL_CreateRenderer(g_win, -1, v && !strcmp(v, "0") ? SDL_RENDERER_ACCELERATED : SDL_RENDERER_PRESENTVSYNC);
    }
    if (!g_ren) g_ren = SDL_CreateRenderer(g_win, -1, SDL_RENDERER_SOFTWARE);
#endif
    if (!g_ren) { fprintf(stderr, "[screen] renderer: %s\n", SDL_GetError()); return -1; }
    SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "0");
    SDL_ShowCursor(SDL_DISABLE);                         /* the game draws its own */
    hle_cursor_hook = warp;
    SDL_RendererInfo ri;
    SDL_GetRendererInfo(g_ren, &ri);
    g_video_thread = pthread_self();
    hle_dd_primary_hook = primary_changed;
    if (SDL_getenv("RA2_D3D_FL_GOT"))                        /* the patched SDL D3D11 (tools/sdl_d3d11_fast.py) */
        fprintf(stderr, "[screen] D3D11 feature level %s%s%s\n", SDL_getenv("RA2_D3D_FL_GOT"),
                SDL_getenv("RA2_D3D_FL") ? ", capped at " : "", SDL_getenv("RA2_D3D_FL") ? SDL_getenv("RA2_D3D_FL") : "");
    fprintf(stderr, "[screen] %s, renderer %s, scaling %s (F12), fullscreen F11%s\n", SDL_GetCurrentVideoDriver(), ri.name,
            k_modes[g_mode], g_confine ? ", mouse held in the picture (" RELEASE_KEYS " lets go)" : "");
    return 0;
}
