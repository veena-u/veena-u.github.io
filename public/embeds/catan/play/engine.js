// The Rust engine compiled to WebAssembly (catan/web): one game at a time, the same calls the Python viewer makes.
export async function loadEngine(bytes) {
  const { instance } = await WebAssembly.instantiate(bytes, {});
  const w = instance.exports, OB = w.observation_bytes();
  const view = () => new Uint8Array(w.memory.buffer, w.buffer(), 4096);
  return {
    observationBytes: OB,
    newGame(seed, players, rules) {
      const ok = w.new_game(seed >>> 0, Math.floor(seed / 2 ** 32) >>> 0, players, rules.points, rules.discardLimit, +rules.friendlyRobber,
                            +rules.balancedDice, +rules.trading, rules.turnBudget);
      if (!ok) throw new Error("unplayable rules");
    },
    toAct: () => { const a = w.to_act(); return a < 0 ? null : a; },
    winner: () => { const a = w.winner(); return a < 0 ? null : a; },
    observe() { if (!w.observe_actor()) return null; const v = view(); return { obs: v.slice(0, OB), mask: Array.from(v.subarray(OB, OB + 257), Boolean) }; },
    observeAs(seat) { if (!w.observe_as(seat)) throw new Error(`no seat ${seat}`); return view().slice(0, OB); },
    step(code, deal) {
      const d = deal || [0, 0, 0, 0, 0];
      if (!w.step(code, deal ? 1 : 0, ...d.map(c => c + 16))) throw new Error(`move ${code} is not legal now`);
    },
    settle: () => w.settle(),
    draft() { if (!w.draft()) return null; const v = view(); return [v[0], Array.from(v.subarray(1, 6)), Array.from(v.subarray(6, 11)), !!v[11]]; },
    forbidden() { const n = w.forbidden(), v = view(); return Array.from({ length: n }, (_, i) => Array.from(v.subarray(5 * i, 5 * i + 5), b => (b << 24) >> 24)); },
    legal() { const n = w.legal(), v = view(); return Array.from({ length: n }, (_, i) => v[2 * i] | v[2 * i + 1] << 8); },
    history(from = 0) {
      const out = [];
      for (;;) {
        const n = w.history(from + out.length), v = view();
        for (let i = 0; i < n; i++) out.push([(v[5 * i] << 24) >> 24, v[5 * i + 1], v[5 * i + 2] | v[5 * i + 3] << 8, v[5 * i + 4]]);
        if (n < 800) return out;
      }
    },
    steps: () => w.history_len(),
  };
}
