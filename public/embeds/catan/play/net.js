// The trained network, in plain JavaScript: the same inputs, transformer, move heads and deal builder as
// train/features.py, train/model.py and train/deals.py, checked number by number against PyTorch (web/test.mjs).

export const SEATS = 4, RES = 5, HEXES = 19, VERTS = 54, EDGES = 72, ROWS = 155, WIDTH = 128, HEADS = 4, DEPTH = 4;
export const ACTIONS = 257, OFFER = 234, COUNTER = 249;
const TRADE_AT = 302, ANSWERS_AT = TRADE_AT + 10, OFFERS_MADE = TRADE_AT + 43, BUDGET_LEFT = TRADE_AT + 44, LAST_AT = TRADE_AT + 45;
const LAST_ACCEPTED = TRADE_AT + 55, LAST_COUNTERED = TRADE_AT + 56, LAST_TRADED = TRADE_AT + 57, RATES_AT = TRADE_AT + 58;
const PROTECTED_AT = TRADE_AT + 63, DICE_AT = PROTECTED_AT + 4, NONE = 255;
const COSTS = [[1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 2, 3], [0, 0, 1, 1, 1]];
const UNLIKELY = -1e9;

export const KIND_RUNS = [
  ["settlement", 0, 54, "corner"], ["city", 54, 54, "corner"], ["road", 108, 72, "edge"], ["robber", 180, 19, "hex"],
  ["pick resource", 199, 5, "resource"], ["bank trade", 204, 20, "resource pair"], ["roll", 224, 1, null], ["end turn", 225, 1, null],
  ["buy card", 226, 1, null], ["play knight", 227, 1, null], ["play road building", 228, 1, null], ["play year of plenty", 229, 1, null],
  ["play monopoly", 230, 1, null], ["rob", 231, 3, "player"], ["offer", 234, 1, null], ["give", 235, 5, "resource"],
  ["ask", 240, 5, "resource"], ["send", 245, 1, null], ["abandon", 246, 1, null], ["accept", 247, 1, null], ["decline", 248, 1, null],
  ["counter", 249, 1, null], ["trade with", 250, 3, "player"], ["take counter", 253, 3, "player"], ["cancel", 256, 1, null]];
export const KIND_OF = new Int32Array(ACTIONS);
KIND_RUNS.forEach(([, start, count], k) => { for (let i = 0; i < count; i++) KIND_OF[start + i] = k; });
const POINTED = KIND_RUNS.map((r, k) => r[3] ? k : -1).filter(k => k >= 0);
export const BANK_PAIRS = [];
for (let g = 0; g < RES; g++) for (let t = 0; t < RES; t++) if (g !== t) BANK_PAIRS.push([g, t]);

// ---- Board tables.
export function boardTables(tables) {
  const hexCorners = tables.hexCorners, edgeEnds = tables.edgeEnds;
  const vertexHexes = Array.from({ length: VERTS }, () => []);
  hexCorners.forEach((cs, h) => cs.forEach(v => vertexHexes[v].push(h)));
  const vertexEdges = Array.from({ length: VERTS }, () => []);
  edgeEnds.forEach(([a, b], e) => { vertexEdges[a].push(e); vertexEdges[b].push(e); });
  const vertexNeighbors = vertexEdges.map((es, v) => es.map(e => edgeEnds[e][0] === v ? edgeEnds[e][1] : edgeEnds[e][0]));
  const edgeNeighbors = edgeEnds.map(([a, b], e) => [...new Set([...vertexEdges[a], ...vertexEdges[b]])].filter(x => x !== e));
  const rel = Uint8Array.from(atob(tables.relations), c => c.charCodeAt(0));
  return { hexCorners, edgeEnds, vertexHexes, vertexEdges, vertexNeighbors, edgeNeighbors, rel };
}

// ---- Inputs (train/features.py).
const bit = (o, start, i) => (o[start + (i >> 3)] >> (i & 7)) & 1;
const onehot = (out, value, classes) => { for (let c = 0; c < classes; c++) out.push(c === value ? 1 : 0); };
const optional = (out, value, classes) => onehot(out, value === NONE ? classes : value, classes + 1);
const pipsOf = t => t === 0 ? 0 : 6 - Math.abs(7 - t);

function pieces(o) {
  const settle = [], city = [], road = [];
  for (let s = 0; s < SEATS; s++) {
    settle.push(Array.from({ length: VERTS }, (_, v) => bit(o, 32 * s, v)));
    city.push(Array.from({ length: VERTS }, (_, v) => bit(o, 32 * s + 8, v)));
    road.push(Array.from({ length: EDGES }, (_, e) => bit(o, 32 * s + 16, e)));
  }
  return { settle, city, road };
}

function rates(o, P) {
  const out = [];
  for (let s = 0; s < SEATS; s++) {
    const has = code => { for (let v = 0; v < VERTS; v++) if ((P.settle[s][v] || P.city[s][v]) && o[166 + v] === code) return true; return false; };
    const generic = has(1);
    out.push(Array.from({ length: RES }, (_, r) => has(2 + r) ? 2 : generic ? 3 : 4));
  }
  return out;
}

export function decode(o, T) {
  const P = pieces(o);
  const buildings = v => [...P.settle.map(s => s[v]), ...P.city.map(c => c[v])];
  const terrain = Array.from({ length: HEXES }, (_, h) => o[128 + h]);
  const tokens = Array.from({ length: HEXES }, (_, h) => o[147 + h]);
  const pips = tokens.map(t => pipsOf(t) / 5);
  const hexes = [], vertices = [], edges = [];
  for (let h = 0; h < HEXES; h++) {
    const f = []; onehot(f, terrain[h], 6); onehot(f, tokens[h], 13); f.push(pips[h]); f.push(o[220] === h ? 1 : 0);
    const sum = new Array(8).fill(0);
    T.hexCorners[h].forEach(v => buildings(v).forEach((b, i) => sum[i] += b));
    hexes.push([...f, ...sum]);
  }
  for (let v = 0; v < VERTS; v++) {
    const f = buildings(v); onehot(f, o[166 + v], 7); f.push(o[300] === v ? 1 : 0);
    const yields = new Array(5).fill(0);
    T.vertexHexes[v].forEach(h => { if (terrain[h] < 5) yields[terrain[h]] += pips[h]; });
    const near = new Array(8).fill(0);
    T.vertexNeighbors[v].forEach(u => buildings(u).forEach((b, i) => near[i] += b));
    const roads = new Array(4).fill(0);
    T.vertexEdges[v].forEach(e => { for (let s = 0; s < SEATS; s++) roads[s] += P.road[s][e]; });
    vertices.push([...f, ...yields, ...near, ...roads]);
  }
  for (let e = 0; e < EDGES; e++) {
    const f = P.road.map(r => r[e]);
    const ends = new Array(8).fill(0);
    T.edgeEnds[e].forEach(v => buildings(v).forEach((b, i) => ends[i] += b));
    const next = new Array(4).fill(0);
    T.edgeNeighbors[e].forEach(x => { for (let s = 0; s < SEATS; s++) next[s] += P.road[s][x]; });
    edges.push([...f, ...ends, ...next]);
  }
  const g = [];
  for (let s = 0; s < SEATS; s++) g.push(s < o[221] ? 1 : 0);
  for (let i = 222; i < 292; i++) g.push(o[i] / 10);
  onehot(g, o[292], 5); onehot(g, o[293], 5); onehot(g, o[294], 4);
  g.push(o[295]); onehot(g, o[296], 16); g.push(o[297]); g.push(o[298] / 4);
  optional(g, o[299], 4); optional(g, o[301], 5);
  for (let i = 0; i < 10; i++) g.push(o[TRADE_AT + i] / 4);
  for (let k = 0; k < 3; k++) onehot(g, Math.min(o[ANSWERS_AT + 11 * k], 4), 5);
  for (let k = 0; k < 3; k++) for (let j = 1; j < 11; j++) g.push(o[ANSWERS_AT + 11 * k + j] / 4);
  g.push(o[OFFERS_MADE] / 10); g.push(o[BUDGET_LEFT] / 40);
  for (let i = 0; i < 10; i++) g.push(o[LAST_AT + i] / 4);
  for (let s = 0; s < SEATS; s++) g.push(bit(o, LAST_ACCEPTED, s));
  for (let s = 0; s < SEATS; s++) g.push(bit(o, LAST_COUNTERED, s));
  g.push(o[LAST_TRADED]);
  for (let r = 0; r < RES; r++) g.push(o[RATES_AT + r] / 4);
  for (let s = 0; s < SEATS; s++) g.push(o[PROTECTED_AT + s]);
  for (let i = 0; i < 11; i++) g.push(o[DICE_AT + i] / 6);
  onehot(g, o[DICE_AT + 11], 13); g.push(o[DICE_AT + 17] / 4);
  return { hexes, vertices, edges, globals: g, P, terrain, tokens };
}

export function players(o, P) {
  const rate = rates(o, P), out = [];
  const mine = Array.from({ length: RES }, (_, r) => o[222 + r] + o[227 + r]);
  const sum = (a, b) => { let t = 0; for (let i = a; i < b; i++) t += o[i]; return t; };
  for (let s = 0; s < SEATS; s++) {
    const k = s - 1;
    const cards = s === 0 ? mine : Array.from({ length: RES }, (_, r) => o[232 + 7 * k + r]);
    const size = s === 0 ? mine.reduce((a, b) => a + b, 0) : o[237 + 7 * k];
    const aside = s === 0 ? sum(227, 232) : o[238 + 7 * k];
    const devs = s === 0 ? sum(258, 263) : o[268 + 2 * k];
    const bought = s === 0 ? sum(263, 268) : o[269 + 2 * k];
    const f = [s === 0 ? 1 : 0, s < o[221] ? 1 : 0, ...cards.map(c => c / 10), size / 10, aside / 10, devs / 5];
    for (let r = 0; r < RES; r++) f.push(s === 0 ? o[258 + r] / 5 : 0);
    f.push(bought, o[280 + s] / 5, o[284 + s] / 15, o[288 + s] / 15, o[294] === s ? 1 : 0, o[292] === s + 1 ? 1 : 0, o[293] === s + 1 ? 1 : 0);
    if (s === 0) for (let i = 0; i < 5; i++) f.push(0); else onehot(f, Math.min(o[ANSWERS_AT + 11 * k], 4), 5);
    for (let j = 1; j < 11; j++) f.push(s === 0 ? 0 : o[ANSWERS_AT + 11 * k + j] / 4);
    f.push(o[299] === s ? 1 : 0, o[PROTECTED_AT + s]);
    for (let r = 0; r < RES; r++) f.push((rate[s][r] - 2) / 2);
    COSTS.forEach(cost => { let m = 0; cost.forEach((c, r) => m += Math.max(c - cards[r], 0)); f.push(m / 5); });
    f.push(o[DICE_AT + 12 + s] / 5, o[DICE_AT + 16] === s + 1 ? 1 : 0);
    out.push(f);
  }
  return out;
}

export function resources(o, P, T, terrain, tokens) {
  const pips = tokens.map((t, h) => h === o[220] ? 0 : pipsOf(t));
  const claims = Array.from({ length: HEXES }, (_, h) => Array.from({ length: SEATS }, (_, s) => {
    let c = 0; T.hexCorners[h].forEach(v => c += P.settle[s][v] + 2 * P.city[s][v]); return c; }));
  const out = [];
  for (let r = 0; r < RES; r++) {
    const f = [(o[222 + r] + o[227 + r]) / 10, o[253 + r] / 19];
    onehot(f, Math.min(Math.max(o[RATES_AT + r], 2), 4) - 2, 3);
    for (let k = 0; k < 3; k++) f.push(o[232 + 7 * k + r] / 10);
    f.push(o[TRADE_AT + r] / 4, o[TRADE_AT + 5 + r] / 4, o[LAST_AT + r] / 4, o[LAST_AT + 5 + r] / 4);
    for (let k = 0; k < 3; k++) f.push(o[ANSWERS_AT + 11 * k + 1 + r] / 4);
    for (let k = 0; k < 3; k++) f.push(o[ANSWERS_AT + 11 * k + 1 + 5 + r] / 4);
    for (let s = 0; s < SEATS; s++) {
      let p = 0; for (let h = 0; h < HEXES; h++) if (terrain[h] === r) p += pips[h] * claims[h][s];
      f.push(p / 10);
    }
    for (let i = 0; i < RES; i++) f.push(i === r ? 1 : 0);
    out.push(f);
  }
  return out;
}

// ---- The heavy arithmetic, run in the engine's WebAssembly module when one is given (several times faster): its
// weights are copied into the module's memory once, and each call's input and output pass through reused arrays.
let K = null;
export function useKernel(kernel) { K = kernel; }
const kHeld = {};
const kArray = (key, n) => { if (!kHeld[key] || kHeld[key].n < n) kHeld[key] = { ptr: K.alloc_f32(n), n }; return kHeld[key].ptr; };
const kView = (ptr, n) => new Float32Array(K.memory.buffer, ptr, n);
function kUpload(t) {
  if (t.kernel !== K) { t.ptr = K.alloc_f32(t.data.length); kView(t.ptr, t.data.length).set(t.data); t.kernel = K; }
  return t.ptr;
}
function kLinear(Wt, bt, x, rows, inp) {
  const n = rows * inp, out = Wt.out, w = kUpload(Wt), b = kUpload(bt), xp = kArray("x", n), yp = kArray("y", rows * out);
  kView(xp, n).set(x.length === n ? x : x.subarray(0, n));
  K.linear(xp, w, b, yp, rows, inp, out);
  return kView(yp, rows * out).slice();
}
function kAttention(qkv, rb, block, rel) {
  const q = kArray("qkv", qkv.length), a = kArray("att", ROWS * WIDTH), bias = kUpload(rb);
  if (!kHeld.rel || kHeld.rel.src !== rel) { const p = K.alloc_f32(Math.ceil(rel.length / 4)); new Uint8Array(K.memory.buffer, p, rel.length).set(rel); kHeld.rel = { ptr: p, src: rel }; }
  kView(q, qkv.length).set(qkv);
  K.attention(q, bias + 4 * block * HEADS * 12, kHeld.rel.ptr, a, ROWS, WIDTH, HEADS);
  return kView(a, ROWS * WIDTH).slice();
}

// ---- Weights: half precision on the wire, each linear layer kept as input × output for fast loops.
const half = h => { const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, f = h & 1023;
  return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15); };

export function loadWeights(manifest, buffer, precision = "f16") {
  const all = precision === "f16" ? Float32Array.from(new Uint16Array(buffer), half) : new Float32Array(buffer);
  const W = {};
  for (const [name, { offset, shape }] of Object.entries(manifest.tensors)) {
    const size = shape.reduce((a, b) => a * b, 1), t = all.subarray(offset, offset + size);
    if (shape.length === 2 && name.endsWith("weight")) {
      const [out, inp] = shape, tr = new Float32Array(size);
      for (let i = 0; i < out; i++) for (let j = 0; j < inp; j++) tr[j * out + i] = t[i * inp + j];
      W[name] = { data: tr, inp, out };
    } else W[name] = { data: t, shape };
  }
  return W;
}

// y = x · W + b for `rows` rows of x (row-major), with W stored input × output.
function linear(W, name, x, rows, inp = W[name + ".weight"].inp) {
  if (K) return kLinear(W[name + ".weight"], W[name + ".bias"], x, rows, inp);
  const { data, out } = W[name + ".weight"], b = W[name + ".bias"].data, y = new Float32Array(rows * out);
  // Four rows at a time, so each weight read from memory is used four times; the sums are kept in doubles and stored
  // once, which is both faster and no less exact than adding into the float32 output.
  const acc = new Float64Array(4 * out);
  let r = 0;
  for (; r + 4 <= rows; r += 4) {
    for (let j = 0; j < out; j++) { const bj = b[j]; acc[j] = bj; acc[out + j] = bj; acc[2 * out + j] = bj; acc[3 * out + j] = bj; }
    const x0 = r * inp, x1 = x0 + inp, x2 = x1 + inp, x3 = x2 + inp;
    for (let i = 0; i < inp; i++) {
      const a0 = x[x0 + i], a1 = x[x1 + i], a2 = x[x2 + i], a3 = x[x3 + i], wo = i * out;
      if (a0 === 0 && a1 === 0 && a2 === 0 && a3 === 0) continue;
      for (let j = 0; j < out; j++) {
        const w = data[wo + j];
        acc[j] += a0 * w; acc[out + j] += a1 * w; acc[2 * out + j] += a2 * w; acc[3 * out + j] += a3 * w;
      }
    }
    y.set(acc, r * out);
  }
  for (; r < rows; r++) {
    const yo = r * out, xo = r * inp;
    y.set(b, yo);
    for (let i = 0; i < inp; i++) {
      const xv = x[xo + i]; if (xv === 0) continue;
      const wo = i * out;
      for (let j = 0; j < out; j++) y[yo + j] += xv * data[wo + j];
    }
  }
  return y;
}

function layerNorm(W, name, x, rows, d = WIDTH) {
  const g = W[name + ".weight"].data, b = W[name + ".bias"].data, y = new Float32Array(rows * d);
  for (let r = 0; r < rows; r++) {
    let mean = 0, v = 0;
    for (let i = 0; i < d; i++) mean += x[r * d + i];
    mean /= d;
    for (let i = 0; i < d; i++) { const t = x[r * d + i] - mean; v += t * t; }
    const inv = 1 / Math.sqrt(v / d + 1e-5);
    for (let i = 0; i < d; i++) y[r * d + i] = (x[r * d + i] - mean) * inv * g[i] + b[i];
  }
  return y;
}

// Exact GELU, as torch's default: x · Φ(x), with erf to about 1e-7.
function erf(x) {
  const s = Math.sign(x), a = Math.abs(x), t = 1 / (1 + 0.5 * a);
  const y = 1 - t * Math.exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return s * y;
}
const gelu = x => {
  if (K) { const p = kArray("gelu", x.length); kView(p, x.length).set(x); K.gelu(p, x.length); x.set(kView(p, x.length)); return x; }
  for (let i = 0; i < x.length; i++) x[i] = 0.5 * x[i] * (1 + erf(x[i] / Math.SQRT2)); return x; };

const flat = rows => { const n = rows.length, d = rows[0].length, out = new Float32Array(n * d); rows.forEach((r, i) => out.set(r, i * d)); return out; };

// ---- The network (train/model.py).
export function forward(W, T, o, mask) {
  const { hexes, vertices, edges, globals, P, terrain, tokens } = decode(o, T);
  const x = new Float32Array(ROWS * WIDTH), pos = W["position"].data, seat = W["seat"].data;
  const put = (row, vals, extra) => { for (let i = 0; i < WIDTH; i++) x[row * WIDTH + i] = vals[i] + extra(i); };
  const sum = linear(W, "embed_global", flat([globals]), 1);
  put(0, sum, i => pos[i]);
  const ppl = linear(W, "embed_player", flat(players(o, P)), SEATS);
  for (let s = 0; s < SEATS; s++) put(1 + s, ppl.subarray(s * WIDTH), i => seat[s * WIDTH + i]);
  const goods = linear(W, "embed_resource", flat(resources(o, P, T, terrain, tokens)), RES);
  for (let r = 0; r < RES; r++) put(5 + r, goods.subarray(r * WIDTH), () => 0);
  const hx = linear(W, "embed_hex", flat(hexes), HEXES), vx = linear(W, "embed_vertex", flat(vertices), VERTS), ex = linear(W, "embed_edge", flat(edges), EDGES);
  for (let h = 0; h < HEXES; h++) put(10 + h, hx.subarray(h * WIDTH), i => pos[(1 + h) * WIDTH + i]);
  for (let v = 0; v < VERTS; v++) put(29 + v, vx.subarray(v * WIDTH), i => pos[(1 + HEXES + v) * WIDTH + i]);
  for (let e = 0; e < EDGES; e++) put(83 + e, ex.subarray(e * WIDTH), i => pos[(1 + HEXES + VERTS + e) * WIDTH + i]);

  const rb = W["relation_bias"].data, D = WIDTH / HEADS, scale = 1 / Math.sqrt(D);
  for (let b = 0; b < DEPTH; b++) {
    const p = `blocks.${b}.`;
    const qkv = linear(W, p + "qkv", layerNorm(W, p + "attend_norm", x, ROWS), ROWS);
    const att = K ? kAttention(qkv, W["relation_bias"], b, T.rel) : new Float32Array(ROWS * WIDTH), row = new Float64Array(ROWS);
    for (let h = 0; h < (K ? 0 : HEADS); h++) {
      const bias = rb.subarray((b * HEADS + h) * 12, (b * HEADS + h) * 12 + 12);
      for (let i = 0; i < ROWS; i++) {
        let best = -Infinity;
        for (let j = 0; j < ROWS; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += qkv[i * 384 + h * D + d] * qkv[j * 384 + WIDTH + h * D + d];
          s = s * scale + bias[T.rel[i * ROWS + j]];
          row[j] = s; if (s > best) best = s;
        }
        let z = 0;
        for (let j = 0; j < ROWS; j++) { row[j] = Math.exp(row[j] - best); z += row[j]; }
        for (let j = 0; j < ROWS; j++) {
          const w = row[j] / z; if (w < 1e-12) continue;
          for (let d = 0; d < D; d++) att[i * WIDTH + h * D + d] += w * qkv[j * 384 + 2 * WIDTH + h * D + d];
        }
      }
    }
    const added = linear(W, p + "out", att, ROWS);
    for (let i = 0; i < x.length; i++) x[i] += added[i];
    const hidden = gelu(linear(W, p + "feed.0", layerNorm(W, p + "feed_norm", x, ROWS), ROWS));
    const back = linear(W, p + "feed.2", hidden, ROWS);
    for (let i = 0; i < x.length; i++) x[i] += back[i];
  }
  const xf = layerNorm(W, "norm", x, ROWS);
  const summary = xf.slice(0, WIDTH);
  const values = Array.from(linear(W, "value", summary, 1), Math.tanh);
  const logProbs = mask ? policy(W, xf, summary, mask) : null;
  return { logProbs, values, summary, goods: xf.slice(5 * WIDTH, 10 * WIDTH) };
}

function policy(W, xf, summary, mask) {
  const rowsOf = { corner: [29, VERTS], edge: [83, EDGES], hex: [10, HEXES], resource: [5, RES], player: [2, 3] };
  const keys = {};
  for (const [name, [at, n]] of Object.entries(rowsOf)) keys[name] = linear(W, "keys", xf.subarray(at * WIDTH, (at + n) * WIDTH), n);
  const questions = linear(W, "questions", summary, 1), gates = W["gates"].data.map(g => 1 / (1 + Math.exp(-g)));
  const point = (q, target) => {
    const [, n] = rowsOf[target], k = keys[target], out = new Float64Array(n);
    for (let t = 0; t < n; t++) { let s = 0; for (let d = 0; d < WIDTH; d++) s += k[t * WIDTH + d] * gates[q * WIDTH + d] * questions[q * WIDTH + d]; out[t] = s / Math.sqrt(WIDTH); }
    return out;
  };
  const scores = new Float64Array(ACTIONS);
  POINTED.forEach((k, q) => {
    const [, start, count, target] = KIND_RUNS[k];
    if (target === "resource pair") {
      const give = point(q, "resource"), get = point(POINTED.length, "resource");
      BANK_PAIRS.forEach(([g, t], i) => scores[start + i] = give[g] + get[t]);
    } else { const s = point(q, target); for (let i = 0; i < count; i++) scores[start + i] = s[i]; }
  });
  const kindLogits = linear(W, "kind", summary, 1);
  for (let a = 0; a < ACTIONS; a++) if (!mask[a]) scores[a] = UNLIKELY;
  const kindLegal = new Array(KIND_RUNS.length).fill(false);
  for (let a = 0; a < ACTIONS; a++) if (mask[a]) kindLegal[KIND_OF[a]] = true;
  const kl = Array.from(kindLogits, (v, k) => kindLegal[k] ? v : UNLIKELY);
  const kmax = Math.max(...kl), kz = Math.log(kl.reduce((s, v) => s + Math.exp(v - kmax), 0)) + kmax;
  const out = new Float64Array(ACTIONS);
  KIND_RUNS.forEach(([, start, count], k) => {
    let m = -Infinity; for (let i = 0; i < count; i++) m = Math.max(m, scores[start + i]);
    let z = 0; for (let i = 0; i < count; i++) z += Math.exp(scores[start + i] - m);
    const lse = Math.log(z) + m;
    for (let i = 0; i < count; i++) out[start + i] = mask[start + i] ? kl[k] - kz + scores[start + i] - lse : UNLIKELY;
  });
  return out;
}

// ---- Deals (train/deals.py and Model.deal_scores).
const STEPS = 9, SIDE = 4, BANK_CARDS = 19, COUNT_OFFSET = 12;
const missing = held => COSTS.map(cost => cost.reduce((m, c, r) => m + Math.max(c - held[r], 0), 0));

export function dealScores(W, summary, goods, counter, counts, hand) {
  const q = linear(W, "deal_query", summary, 1), ce = W["count_embed"].data, h = new Float32Array(WIDTH);
  for (let i = 0; i < WIDTH; i++) h[i] = q[(counter ? WIDTH : 0) + i];
  for (let r = 0; r < RES; r++) { const at = (r * 25 + counts[r] + COUNT_OFFSET) * WIDTH; for (let i = 0; i < WIDTH; i++) h[i] += ce[at + i]; }
  const after = hand.map((c, r) => Math.max(c + counts[r], 0)), mA = missing(after), mH = missing(hand);
  const whole = [...mA.map(m => m / 5), ...mH.map((m, i) => (m - mA[i]) / 5)];
  const stopIn = new Float32Array(WIDTH + 8); stopIn.set(h); stopIn.set(whole, WIDTH);
  const stop = linear(W, "stop_out", gelu(linear(W, "stop_hidden", stopIn, 1)), 1)[0];
  const both = new Float32Array(RES * 266);
  for (let r = 0; r < RES; r++) {
    both.set(h, r * 266); both.set(goods.subarray(r * WIDTH, (r + 1) * WIDTH), r * 266 + WIDTH);
    both.set([after[r] / 10, counts[r] / 4], r * 266 + 256); both.set(whole, r * 266 + 258);
  }
  const cards = linear(W, "card_out", gelu(linear(W, "card_hidden", both, RES)), RES);
  return { stop, cards: [0, 1, 2, 3, 4].map(r => cards[2 * r]).concat([0, 1, 2, 3, 4].map(r => cards[2 * r + 1])) };
}

export function stepMask(counts, hand, forbidden) {
  const gave = counts.some(c => c < 0), asked = counts.some(c => c > 0);
  const same = forbidden.some(f => f.every((c, r) => c === counts[r]));
  const valid = gave && asked && !same;
  const gives = counts.reduce((s, c) => s + Math.max(-c, 0), 0), asks = counts.reduce((s, c) => s + Math.max(c, 0), 0);
  let placed = 0;
  counts.forEach((c, r) => { placed = Math.max(placed, c < 0 ? r + 1 : c > 0 ? 6 + r : 0); });
  const mask = [valid];
  for (let r = 0; r < RES; r++) mask.push(counts[r] <= 0 && -counts[r] < hand[r] && gives < SIDE && r + 1 >= placed);
  for (let r = 0; r < RES; r++) mask.push(counts[r] >= 0 && counts[r] < BANK_CARDS - hand[r] && asks < SIDE && gave && 6 + r >= placed);
  if (!mask.some(Boolean)) mask[0] = true;
  return mask;
}

const logSigmoid = x => x >= 0 ? -Math.log1p(Math.exp(-x)) : x - Math.log1p(Math.exp(x));
export function stepLogProbs(stop, cards, mask) {
  const canGo = mask.slice(1).some(Boolean);
  const c = cards.map((v, i) => mask[i + 1] ? v : UNLIKELY), m = Math.max(...c), z = Math.log(c.reduce((s, v) => s + Math.exp(v - m), 0)) + m;
  const stopLp = mask[0] && canGo ? logSigmoid(stop) : mask[0] ? 0 : UNLIKELY;
  const goLp = canGo && mask[0] ? logSigmoid(-stop) : canGo ? 0 : UNLIKELY;
  return [stopLp, ...c.map(v => goLp + v - z)];
}

function sample(logProbs, temperature, random) {
  const scaled = Array.from(logProbs, v => v / temperature), m = Math.max(...scaled);
  const p = scaled.map(v => Math.exp(v - m)), z = p.reduce((a, b) => a + b, 0);
  let u = random() * z;
  for (let i = 0; i < p.length; i++) { u -= p[i]; if (u <= 0) return i; }
  return p.findLastIndex(x => x > 0);
}

// One move for the player to act, as train/agents.Network plays it, plus the deal when it offers or counters.
export function act(W, T, o, mask, forbidden, { temperature = 0.8, random = Math.random } = {}) {
  const { logProbs, summary, goods } = forward(W, T, o, mask);
  const action = sample(logProbs, temperature, random);
  if (action !== OFFER && action !== COUNTER) return { action };
  const hand = [0, 1, 2, 3, 4].map(r => o[222 + r]), counts = [0, 0, 0, 0, 0];
  for (let k = 0; k < STEPS; k++) {
    const mask_ = stepMask(counts, hand, forbidden);
    const { stop, cards } = dealScores(W, summary, goods, action === COUNTER, counts, hand);
    const pick = sample(stepLogProbs(stop, cards, mask_), temperature, random);
    if (pick === 0) break;
    const card = pick - 1;
    counts[card % RES] += card < RES ? -1 : 1;
  }
  return { action, deal: counts };
}
