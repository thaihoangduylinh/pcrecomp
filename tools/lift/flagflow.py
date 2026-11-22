"""Which flags cross calls and returns (PLAN.md 4.3, Red Alert 2 port).

The lifted C hands the four flag variables over at every transfer: each call
stores them for the callee (RECOMP_FLAGS_OUT) and each `ret` stores them for
the caller. Stores to globals can never be dropped, so every flag computation
before a call or a return stays alive, and those are most of the lifted
instructions. Almost no code reads flags across a call: x86 compilers never
do, and hand-written assembly rarely.

This works out, over the whole program, where flags really do cross:

  reads_in   functions whose entry flags are read (before written) on some
             path: their callers must store the flags before calling;
  out_needed functions some caller reads flags from after the call returns:
             their `ret` must store them.

Everything else can skip the stores, and the C compiler then drops the flag
computations nobody reads.

Liveness runs backward over each body's instructions with two bits, CF and
the rest (ZF, SF, OF, PF: the lifter keeps those in one lazy state). Calls,
tail jumps and returns join the bodies: a call reads all flags when its callee
is in reads_in and writes all of them (the callee's), a `ret` reads them when
the function is in out_needed, a tail jump when its target is in reads_in
(and the target returns to this function's callers). A function that hands
its entry flags straight back (an empty one) ends up in reads_in through its
own `ret`. Anything not known (an indirect jump, an unknown instruction's
reads) counts as reading everything. Both sets only grow, from empty, until
nothing changes: the least fixed point.

Transfers to code not known here (an indirect call or jump, a function with
no body): `icall_reads` says whether such a target may read the flags it is
handed (True: as many as everything; False: compiled code never hands flags
over an indirect transfer); `icall_live` (an output) lists indirect call
sites whose flags are read after the call (their callees, whichever they
are, must store them).
"""

C, Z = 1, 2
ALL = C | Z

_COND = {   # condition suffix -> the flags it reads
    'a': C | Z, 'nbe': C | Z, 'be': C | Z, 'na': C | Z,
    'b': C, 'nae': C, 'c': C, 'ae': C, 'nb': C, 'nc': C,
    'e': Z, 'z': Z, 'ne': Z, 'nz': Z, 's': Z, 'ns': Z, 'o': Z, 'no': Z,
    'p': Z, 'pe': Z, 'np': Z, 'po': Z, 'l': Z, 'nge': Z, 'ge': Z, 'nl': Z,
    'le': Z, 'ng': Z, 'g': Z, 'nle': Z,
    'u': Z, 'nu': Z,          # fcmovu / fcmovnu (PF)
}
_KILL_ALL = {'add', 'sub', 'cmp', 'neg', 'adc', 'sbb', 'and', 'or', 'xor', 'test', 'imul', 'mul',
             'div', 'idiv', 'cmpxchg', 'cmpxchg8b', 'xadd', 'sahf', 'popf', 'popfd', 'bt', 'bts',
             'btr', 'btc', 'bsf', 'bsr', 'fcomi', 'fcomip', 'fucomi', 'fucomip', 'lzcnt', 'tzcnt',
             'popcnt', 'daa', 'das', 'aaa', 'aas', 'aam', 'aad', 'cmpsb', 'cmpsw', 'cmpsd',
             'scasb', 'scasw', 'scasd', 'ptest', 'comiss', 'comisd', 'ucomiss', 'ucomisd'}
_READ = {'adc': C, 'sbb': C, 'rcl': C, 'rcr': C, 'cmc': C, 'salc': C,
         'pushf': ALL, 'pushfd': ALL, 'lahf': ALL, 'into': Z,
         'loope': Z, 'loopz': Z, 'loopne': Z, 'loopnz': Z}


def _imm(op_str):
    """The last operand as an immediate, or None."""
    last = op_str.split(',')[-1].strip()
    try:
        return int(last, 0)
    except ValueError:
        return None


def flag_effect(mnemonic, op_str):
    """(reads, kills) as C/Z bits for one instruction."""
    parts = mnemonic.split()
    m = parts[-1]
    rep = len(parts) > 1 and parts[0].startswith('rep')
    reads = _READ.get(m, 0)
    if m.startswith('j') and m not in ('jmp', 'jecxz', 'jcxz'):
        reads |= _COND.get(m[1:], ALL)
    elif m.startswith('set'):
        reads |= _COND.get(m[3:], ALL)
    elif m.startswith('fcmov'):
        reads |= _COND.get(m[5:], ALL)
    elif m.startswith('cmov'):
        reads |= _COND.get(m[4:], ALL)
    kills = 0
    if m in _KILL_ALL:
        kills = 0 if rep else ALL                      # a rep with ecx 0 leaves them
    elif m in ('inc', 'dec'):
        kills = Z
    elif m in ('clc', 'stc', 'cmc'):
        kills = C
    elif m in ('shl', 'sal', 'shr', 'sar', 'shld', 'shrd'):
        n = _imm(op_str)                               # by cl (maybe 0): unchanged
        kills = ALL if n is not None and n & 31 else 0
        if n is None and ',' not in op_str:            # the one-operand form: by 1
            kills = ALL
    elif m in ('rol', 'ror', 'rcl', 'rcr'):
        n = _imm(op_str)
        kills = C if (n is not None and n & 31) or ',' not in op_str else 0
    return reads, kills


class Body:
    """One function's instructions as the analysis needs them.

    `insns`: (address, end, mnemonic, op_str, branch target or None) in the
    lifted order; `gap`: indices whose fall-through is not the next one."""
    __slots__ = ('addr', 'entry', 'idx', 'n', 'succ', 'exits', 'reads', 'kills', 'calls', 'pushjmp')

    def __init__(self, addr, insns, imports=()):
        self.addr = addr
        self.n = len(insns)
        self.idx = {a: i for i, (a, *_rest) in enumerate(insns)}
        self.entry = self.idx.get(addr, 0)   # code can sit below the entry
        self.succ = [[] for _ in insns]
        self.exits = [[] for _ in insns]    # ('ret',) ('tail', va) ('unk',)
        self.reads = [0] * self.n
        self.kills = [0] * self.n
        self.calls = {}                     # index -> callee va, or None (indirect), or 'imp'
        self.pushjmp = set()                # targets reached by `push label; jmp target`: they return here
        for i, (a, end, m, ops, tgt) in enumerate(insns):
            base = m.split()[-1]
            r, k = flag_effect(m, ops)
            self.reads[i], self.kills[i] = r, k
            nxt = i + 1 if i + 1 < self.n and insns[i + 1][0] == end else None

            def flow(va):
                j = self.idx.get(va)
                if j is not None:
                    self.succ[i].append(j)
                else:
                    self.exits[i].append(('tail', va))

            if base in ('ret', 'retn', 'retf', 'iret', 'iretd'):
                self.exits[i].append(('ret',))
                continue
            if base == 'jmp':
                if tgt is None:
                    self.exits[i].append(('unk',))   # a switch: its arms are not followed
                else:
                    if i > 0 and insns[i - 1][2] == 'push' and tgt not in self.idx:
                        self.pushjmp.add(tgt)
                        lbl = _imm(insns[i - 1][3])
                        if lbl is not None and lbl in self.idx:
                            self.succ[i].append(self.idx[lbl])
                    flow(tgt)
                continue
            if base in ('int3', 'hlt', 'ud2'):
                continue
            if base == 'call':
                self.calls[i] = 'imp' if tgt is not None and tgt in imports else tgt
                self.kills[i] = ALL
            elif (m.startswith('j') or m.startswith('loop')) and tgt is not None:
                flow(tgt)
            if nxt is not None:
                self.succ[i].append(nxt)
            else:
                flow(end)                                # off the end, or past a gap: a tail


def analyze(bodies, icall_reads=True, max_rounds=100):
    """bodies: {va: Body}. Returns (reads_in, out_needed, icall_live, rounds)."""
    reads_in, out_needed = set(), set()
    icall_live = set()                                   # (function va, call index)
    preds = {}
    for b in bodies.values():
        p = [[] for _ in range(b.n)]
        for i, ss in enumerate(b.succ):
            for j in ss:
                p[j].append(i)
        preds[b.addr] = p
    callers_of = {}                                      # va -> function vas that call or jump to it
    for b in bodies.values():
        for i, t in b.calls.items():
            if isinstance(t, int):
                callers_of.setdefault(t, set()).add(b.addr)
        for ex in b.exits:
            for e in ex:
                if e[0] == 'tail':
                    callers_of.setdefault(e[1], set()).add(b.addr)
    for b in bodies.values():                            # `push label; jmp f`: f returns to us
        out_needed |= {t for t in b.pushjmp if t in bodies}

    def solve(b):
        """Liveness over one body; returns (entry live, {call index: live after})."""
        live_in = [0] * b.n
        work = list(range(b.n - 1, -1, -1))
        on = [True] * b.n
        after = {}
        p = preds[b.addr]
        while work:
            i = work.pop()
            on[i] = False
            out = 0
            for j in b.succ[i]:
                out |= live_in[j]
            for e in b.exits[i]:
                if e[0] == 'ret':
                    out |= ALL if b.addr in out_needed else 0
                elif e[0] == 'tail':
                    out |= ALL if (e[1] in reads_in if e[1] in bodies else icall_reads) else 0
                else:                                    # an indirect jump: an unknown target
                    out |= ALL if icall_reads else 0
            t = b.calls.get(i, 0)
            if i in b.calls:
                after[i] = out
                if t == 'imp':
                    r = 0
                elif t is None:
                    r = ALL if icall_reads else 0
                else:
                    r = ALL if (t in reads_in if t in bodies else icall_reads) else 0
                v = r
            else:
                v = b.reads[i] | (out & ~b.kills[i])
            if v != live_in[i]:
                live_in[i] = v
                for j in p[i]:
                    if not on[j]:
                        on[j] = True
                        work.append(j)
        return (live_in[b.entry] if b.n else 0), after

    dirty = set(bodies)
    rounds = 0
    while dirty and rounds < max_rounds:
        rounds += 1
        nxt = set()
        for va in sorted(dirty):
            b = bodies[va]
            entry, after = solve(b)
            if entry and va not in reads_in:
                reads_in.add(va)
                nxt |= callers_of.get(va, set())
            for i, live in after.items():
                t = b.calls[i]
                if not live:
                    continue
                if t is None:
                    icall_live.add((va, i))
                elif isinstance(t, int) and t in bodies and t not in out_needed:
                    out_needed.add(t)
                    nxt.add(t)
            # a tail jump returns to this function's callers: its target returns for us
            if va in out_needed:
                for ex in b.exits:
                    for e in ex:
                        if e[0] == 'tail' and e[1] in bodies and e[1] not in out_needed:
                            out_needed.add(e[1])
                            nxt.add(e[1])
        dirty = nxt
    if dirty:
        raise RuntimeError('flagflow: no fixed point after %d rounds' % max_rounds)
    return reads_in, out_needed, icall_live, rounds


def _selftest():
    def body(addr, rows):
        return Body(addr, rows)
    # f: cmp; call g; jz  -> g must hand flags back; g = ret (passes f's in) -> g reads in too
    f = body(0x100, [(0x100, 0x102, 'cmp', 'eax, 1', None), (0x102, 0x107, 'call', '0x200', 0x200),
                     (0x107, 0x109, 'je', '0x10b', 0x10b), (0x109, 0x10a, 'ret', '', None),
                     (0x10b, 0x10c, 'ret', '', None)])
    g = body(0x200, [(0x200, 0x201, 'ret', '', None)])
    # h: xor; call k; ret  -> nothing crosses
    h = body(0x300, [(0x300, 0x302, 'xor', 'eax, eax', None), (0x302, 0x307, 'call', '0x400', 0x400),
                     (0x307, 0x308, 'ret', '', None)])
    k = body(0x400, [(0x400, 0x402, 'add', 'eax, 1', None), (0x402, 0x403, 'ret', '', None)])
    # m: entry reads CF (sbb) -> reads_in
    m = body(0x500, [(0x500, 0x502, 'sbb', 'eax, eax', None), (0x502, 0x503, 'ret', '', None)])
    # t: tail jump into m from a function whose caller reads nothing
    t = body(0x600, [(0x600, 0x605, 'jmp', '0x500', 0x500)])
    ri, on, il, _ = analyze({b.addr: b for b in (f, g, h, k, m, t)})
    assert 0x200 in on and 0x200 in ri, (ri, on)
    assert 0x400 not in on and 0x400 not in ri and 0x300 not in ri
    assert 0x500 in ri and 0x600 in ri and 0x100 not in ri
    assert flag_effect('inc', 'eax') == (0, Z) and flag_effect('shl', 'eax, cl') == (0, 0)
    assert flag_effect('ja', '0x10') == (ALL, 0) and flag_effect('repe cmpsb', '') == (0, 0)
    print('flagflow selftest ok')


if __name__ == '__main__':
    _selftest()
