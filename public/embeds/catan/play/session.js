// One game against the network, in the browser: a port of the Python viewer's Session (catan/viewer/__main__.py).
// It plays the network's seats, keeps the running story and the trade board, and builds the view the page draws.
import { act as think, forward, boardTables, KIND_RUNS, KIND_OF, BANK_PAIRS, OFFER, COUNTER } from "./net.js";

const RESOURCES = ["brick", "lumber", "wool", "grain", "ore"];
const TERRAINS = ["hills", "forest", "pasture", "fields", "mountains", "desert"];
const DEV_CARDS = ["knight", "victory point", "road building", "year of plenty", "monopoly"];
const PHASES = ["place settlement", "place road", "turn", "dice", "discard", "move robber", "steal", "draw", "road building",
  "year of plenty", "monopoly", "over", "choose victim", "respond", "confirm", "draft"];
export const NAMES = ["Blue", "Orange", "Green", "Pink"];
const END_TURN = 225, SEND = 245, ACCEPT = 247, DECLINE = 248, TRADE_WITH = 250, TAKE_COUNTER = 253, CANCEL = 256;
const PHASE_AT = 296, TRADE_AT = 302, ANSWERS_AT = TRADE_AT + 10, OFFERS_MADE = TRADE_AT + 43, BUDGET_LEFT = TRADE_AT + 44;
const LAST_AT = TRADE_AT + 45, LAST_TRADED = TRADE_AT + 57, RATES_AT = TRADE_AT + 58, PROTECTED_AT = TRADE_AT + 63;

const bitset = (o, start, length, count) => {
  const out = [];
  for (let i = 0; i < count && i < 8 * length; i++) if ((o[start + (i >> 3)] >> (i & 7)) & 1) out.push(i);
  return out;
};
const sides = counts => [counts.map(c => Math.max(-c, 0)), counts.map(c => Math.max(c, 0))];
const NAMED = ["Brick", "Lumber", "Wool", "Grain", "Ore"];
const listed = counts => counts.flatMap((n, r) => n ? [`${n} ${NAMED[r]}`] : []).join(", ");
const cards = counts => { const named = counts.flatMap((n, r) => n ? [`${n} ${RESOURCES[r]}`] : []); return named.length ? named.join(", ") : "anything"; };

export class Session {
  constructor(engine, weights, tables, config) {
    Object.assign(this, { engine, weights, tables, config });
    this.geometry = { hexes: tables.hexPoints, vertices: tables.vertexPoints, edges: tables.edgeEnds, corners: tables.hexCorners, coast: tables.coast };
    this.board = boardTables(tables);
  }

  new(seed, seat) {
    const n = this.config.players;
    this.seed = seed ?? Math.floor(Math.random() * 2 ** 31);
    this.human = seat ?? Math.floor(Math.random() * n);
    this.engine.newGame(this.seed, n, this.config.rules);
    this.engine.settle();
    this.events = []; this.read = 0; this.lastRoll = null; this.victim = null; this.mover = 0; this.trade = null;
    // Each payout (a roll, or starting resources), for the page to animate: which hexes paid whom.
    this.payouts = []; this.placed = Array(this.config.players).fill(0);
  }

  absolute(relative) { return (this.human + relative) % this.config.players; }

  act(code, deal) {
    if (this.engine.toAct() !== this.human || !this.engine.legal().includes(code)) throw new Error("that move is not yours to make now");
    if (deal && code !== OFFER && code !== COUNTER) throw new Error("a deal goes with an offer or a counter-offer");
    this.take(code, deal || null);
  }

  // The network plays the seat to act, as train/agents.Network does.
  step() {
    const actor = this.engine.toAct();
    if (actor === null || actor === this.human) return;
    const { obs, mask } = this.engine.observe();
    const move = think(this.weights, this.board, obs, mask, this.engine.forbidden(), { temperature: this.config.temperature });
    // With search on, as train/search.Searcher plays: the network still makes its own offers and counter-offers,
    // and every other move it can look ahead on is the one the search picks.
    if (this.config.search && move.action !== OFFER && move.action !== COUNTER && mask.filter(Boolean).length > 1) {
      const picked = this.lookAhead(this.config.search);
      if (picked >= 0) return this.take(picked, null);
    }
    this.take(move.action, move.deal || null);
  }

  // A short search from the current position: the engine imagines the hidden cards and plays out a few lines, and
  // the network judges every position it reaches.
  lookAhead({ simulations, worlds, considered }) {
    const e = this.engine;
    e.searchStart(simulations, worlds, considered, Math.floor(Math.random() * 2 ** 52));
    for (let n; (n = e.searchLeaves()) > 0;) {
      const logProbs = new Float32Array(n * 257), values = new Float32Array(n * 4);
      for (let i = 0; i < n; i++) {
        const { obs, mask } = e.searchLeaf(i), out = forward(this.weights, this.board, obs, mask);
        logProbs.set(out.logProbs, i * 257); values.set(out.values, i * 4);
      }
      e.searchExpand(logProbs, values);
    }
    return e.searchResult();
  }

  take(code, deal) {
    const e = this.engine, actor = e.toAct(), phase = PHASES[e.observeAs(this.human)[PHASE_AT]];
    this.countering = null;
    this.track(actor, phase, code, deal, e.draft());
    e.step(code, deal);
    if (this.countering && this.trade) {
      const seat = this.countering[0], proposer = this.trade.proposer;
      const relative = (seat - proposer + this.config.players) % this.config.players;
      if (e.observeAs(proposer)[ANSWERS_AT + 11 * (relative - 1)] === 4) this.answer(seat, "countered", ...this.countering.slice(1));
    }
    this.countering = null;
    e.settle();
    this.narrate();
    this.settled();
  }

  say(seat, text) { this.events.push({ seat, text }); }

  track(actor, phase, code, deal, drafted) {
    const n = this.config.players;
    if (code === END_TURN) this.trade = null;
    if (code === OFFER && deal) this.openOffer(actor, ...sides(deal));
    else if (phase === "draft" && code === SEND && drafted) this.openOffer(actor, drafted[1], drafted[2]);
    else if (phase === "respond" && this.trade) {
      if (code === ACCEPT) this.answer(actor, "accepted");
      else if (code === DECLINE) this.answer(actor, "declined");
      else if (code === COUNTER && deal) this.countering = [actor, ...sides(deal)];
      else if (code === SEND && drafted && drafted[0] === 2) this.answer(actor, "countered", drafted[1], drafted[2]);
    } else if (phase === "confirm" && this.trade) {
      if (code >= TRADE_WITH && code < TAKE_COUNTER) { const p = (actor + code - TRADE_WITH + 1) % n; this.close(`traded with ${NAMES[p]}`, p); }
      else if (code >= TAKE_COUNTER && code < CANCEL) { const p = (actor + code - TAKE_COUNTER + 1) % n; this.close(`took ${NAMES[p]}'s counter-offer`, p); }
      else if (code === CANCEL) this.close("called off the trade");
    }
  }

  openOffer(proposer, give, get) {
    const n = this.config.players, answers = {};
    for (let k = 1; k < n; k++) { const s = (proposer + k) % n; answers[s] = { seat: s, status: "waiting", counter: null }; }
    this.trade = { proposer, give, get, result: null, partner: null, answers };
    this.say(proposer, `offered ${cards(give)} for ${cards(get)}`);
  }

  answer(seat, status, give, get) {
    const entry = this.trade.answers[seat];
    entry.status = status;
    if (give) { entry.counter = { give, get }; this.say(seat, `declined, and offered ${cards(give)} for ${cards(get)} instead`); }
    else this.say(seat, `${status} the offer`);
  }

  close(text, partner = null) { this.trade.result = text; this.trade.partner = partner; this.say(this.trade.proposer, text); }

  settled() {
    if (!this.trade || this.trade.result) return;
    const phase = PHASES[this.engine.observeAs(this.human)[PHASE_AT]];
    if (["draft", "respond", "confirm"].includes(phase)) return;
    const answers = Object.values(this.trade.answers);
    answers.forEach(a => { if (a.status === "waiting") a.status = "declined"; });
    this.close(answers.some(a => a.status === "countered") ? "couldn't take the counter-offer, so the trade fell through" : "got no takers");
  }

  narrate() {
    const history = this.engine.history(this.read);
    history.forEach(([actor, phase, code, current], k) => {
      // Chance (a roll, a steal, a card drawn) belongs to the player whose turn it is.
      if (actor < 0) this.mover = current;
      const text = this.describe(actor, phase, code, this.read + k);
      const event = text ? { seat: actor >= 0 ? actor : this.mover, text } : null;
      if (event) this.events.push(event);
      if (this.pending) { if (event) event.details = this.pending; else this.events.push(...this.pending); this.pending = null; }
      if (actor >= 0) this.mover = actor;
    });
    this.read += history.length;
  }

  describe(actor, phase, code, index) {
    const n = this.config.players;
    if (actor < 0) {
      const mover = this.mover;
      if (PHASES[phase] === "dice") { this.lastRoll = code + 2; this.collect(code + 2, index); return `rolled ${code + 2}`; }
      if (PHASES[phase] === "steal") {
        let victim = this.victim;
        if (victim === null && n === 2) victim = 1 - mover;
        this.victim = null;
        if (victim === null) return "stole a card";
        return [mover, victim].includes(this.human) ? `stole 1 ${RESOURCES[code]} from ${NAMES[victim]}` : `stole a card from ${NAMES[victim]}`;
      }
      if (PHASES[phase] === "draw") return mover === this.human ? `bought a ${DEV_CARDS[code]}` : "bought a development card";
      return null;
    }
    const [kind, start] = KIND_RUNS[KIND_OF[code]], i = code - start;
    if (PHASES[phase] === "place settlement" && kind === "settlement") {
      this.placed[actor] += 1;
      if (this.placed[actor] === 2) {
        const o = this.engine.observeAs(this.human), hexes = this.tables.hexCorners.map((cs, h) => cs.includes(i) && o[128 + h] <= 4 ? h : -1).filter(h => h >= 0);
        const got = (this.engine.gains(index) || [])[actor] || [0, 0, 0, 0, 0];
        if (got.some(Boolean)) {
          this.pending = [{ seat: actor, text: `received ${listed(got)}` }];
          this.payouts.push({ id: this.payouts.length + 1, hexes: [], cards: hexes.map(h => ({ seat: actor, hex: h, resource: o[128 + h], count: 1 })) });
        }
      }
      return "placed a Settlement";
    }
    if (PHASES[phase] === "place road" && kind === "road") return "placed a Road";
    if (["settlement", "city", "road"].includes(kind)) return `built a ${kind === "city" ? "City" : kind === "road" ? "Road" : "Settlement"}`;
    if (kind === "robber") return "moved the robber";
    if (kind === "rob") { this.victim = (actor + i + 1) % n; return null; }
    if (kind === "pick resource") return { discard: "discarded a card", "year of plenty": `took 1 ${RESOURCES[i]} from the bank`, monopoly: `called a monopoly on ${RESOURCES[i]}` }[PHASES[phase]] || null;
    if (kind === "bank trade") { const [g, t] = BANK_PAIRS[i]; return `traded ${RESOURCES[g]} for 1 ${RESOURCES[t]} with the bank`; }
    if (kind.startsWith("play ")) return `played ${kind.slice(5).replace(/\b\w/g, c => c.toUpperCase())}`;
    if (kind === "end turn") return "ended their turn";
    return null;
  }

  // Who collected what on a roll: the cards the engine actually paid each seat (production is public, so every
  // player's take is told). The board only says which hexes they came from, for the animation.
  collect(sum, index) {
    if (sum === 7) { this.pending = []; return; }
    const o = this.engine.observeAs(this.human), n = this.config.players, corners = this.tables.hexCorners;
    const got = (this.engine.gains(index) || []).slice(0, n).map(g => g.slice());
    const rolled = [], cards = [];
    for (let h = 0; h < 19; h++) if (o[147 + h] === sum && h !== o[220] && o[128 + h] <= 4) rolled.push(h);
    const left = got.map(g => g.slice());
    // Each seat's cards of a resource, from the rolled hexes of that resource next to its buildings first.
    for (const h of rolled) for (let s = 0; s < n; s++) {
      const seat = this.absolute(s), r = o[128 + h];
      let k = 0;
      corners[h].forEach(v => { const bit = at => (o[at + (v >> 3)] >> (v & 7)) & 1; k += bit(32 * s) + 2 * bit(32 * s + 8); });
      const count = Math.min(k, left[seat][r]);
      if (count > 0) { cards.push({ seat, hex: h, resource: r, count }); left[seat][r] -= count; }
    }
    left.forEach((g, seat) => g.forEach((c, r) => { const h = rolled.find(x => o[128 + x] === r); if (c > 0 && h != null) cards.push({ seat, hex: h, resource: r, count: c }); }));
    this.pending = got.map((g, s) => g.some(Boolean) ? { seat: s, text: `received ${listed(g)}` } : null).filter(Boolean);
    if (!this.pending.length) this.pending = [{ seat: null, text: "no one received anything" }];
    this.payouts.push({ id: this.payouts.length + 1, hexes: rolled, cards });
  }

  // What the network makes of the position, seen from your seat: its forecast of each player's chance of winning, and
  // on your turn the move it would make (with its chance), so you can compare.
  insight() {
    const e = this.engine, key = `${e.steps()}:${e.toAct()}`;
    if (this.insightKey === key) return this.insightValue;
    const n = this.config.players, mine = e.toAct() === this.human;
    const o = mine ? e.observe().obs : e.observeAs(this.human), mask = mine ? e.observe().mask : null;
    const out = forward(this.weights, this.board, o, mask);
    // The forecast is the expected reward: +1 for a win, −1/(n−1) otherwise; turned back into a chance of winning.
    const loss = 1 / (n - 1), chance = out.values.slice(0, n).map(v => Math.min(1, Math.max(0, (v + loss) / (1 + loss))));
    const total = chance.reduce((a, b) => a + b, 0) || 1, forecast = Array(n).fill(0);
    chance.forEach((c, s) => { forecast[this.absolute(s)] = c / total; });
    let hint = null;
    if (mine && out.logProbs) {
      let best = -1; for (let a = 0; a < out.logProbs.length; a++) if (mask[a] && (best < 0 || out.logProbs[a] > out.logProbs[best])) best = a;
      const move = this.moves().find(m => m.code === best);
      if (move) hint = { ...move, chance: Math.exp(out.logProbs[best]) };
    }
    this.insightKey = key; this.insightValue = { forecast, hint };
    return this.insightValue;
  }

  moves() {
    if (this.engine.toAct() !== this.human) return [];
    const n = this.config.players;
    return this.engine.legal().map(code => {
      const [name, start, , target] = KIND_RUNS[KIND_OF[code]], i = code - start, move = { code, kind: name };
      if (target === "corner") move.vertex = i;
      else if (target === "edge") move.edge = i;
      else if (target === "hex") move.hex = i;
      else if (target === "resource") move.resource = RESOURCES[i];
      else if (target === "player") move.seat = (this.human + i + 1) % n;
      else if (target === "resource pair") { move.give = RESOURCES[BANK_PAIRS[i][0]]; move.get = RESOURCES[BANK_PAIRS[i][1]]; }
      return move;
    });
  }

  view() {
    const e = this.engine, o = Array.from(e.observeAs(this.human)), n = this.config.players, rules = this.config.rules;
    const seatOf = relative => this.absolute(relative), settlements = [], cities = [], roads = [];
    for (let s = 0; s < n; s++) {
      const base = 32 * s;
      bitset(o, base, 8, 54).forEach(v => settlements.push({ vertex: v, seat: seatOf(s) }));
      bitset(o, base + 8, 8, 54).forEach(v => cities.push({ vertex: v, seat: seatOf(s) }));
      bitset(o, base + 16, 16, 72).forEach(x => roads.push({ edge: x, seat: seatOf(s) }));
    }
    const holder = value => value === 0 ? null : seatOf(value - 1);
    const sum = (a, b) => o.slice(a, b).reduce((x, y) => x + y, 0);
    const seats = [];
    for (let s = 0; s < n; s++) {
      const hand = s === 0 ? sum(222, 227) : o[232 + 7 * (s - 1) + 5], devs = s === 0 ? sum(258, 263) : o[268 + 2 * (s - 1)];
      const seat = { seat: seatOf(s), name: NAMES[seatOf(s)], you: s === 0, points: o[288 + s], cards: hand, devCards: devs, knights: o[280 + s],
        roadLength: o[284 + s], longestRoad: holder(o[292]) === seatOf(s), largestArmy: holder(o[293]) === seatOf(s), protected: !!o[PROTECTED_AT + s] };
      if (s > 0) {
        const at = ANSWERS_AT + 11 * (s - 1), counter = o.slice(at + 1, at + 11);
        seat.answer = ["", "waiting", "accepted", "declined", "countered"][Math.min(o[at], 4)];
        seat.counter = counter.some(Boolean) ? { give: counter.slice(0, 5), get: counter.slice(5) } : null;
      }
      seats.push(seat);
    }
    const offer = o.slice(TRADE_AT, TRADE_AT + 10), last = o.slice(LAST_AT, LAST_AT + 10);
    let [drafting, give, get, closed] = e.draft() || [0, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], false];
    if (e.toAct() !== this.human) { drafting = 0; give = [0, 0, 0, 0, 0]; get = [0, 0, 0, 0, 0]; }
    const trade = this.trade && { ...this.trade, answers: Object.values(this.trade.answers).sort((a, b) => a.seat - b.seat) };
    return {
      seed: this.seed, players: n, human: this.human, toAct: e.toAct(), winner: e.winner(), geometry: this.geometry,
      rules: { points: rules.points, discardLimit: rules.discardLimit, friendlyRobber: rules.friendlyRobber, balancedDice: rules.balancedDice, trading: rules.trading },
      board: {
        terrain: o.slice(128, 147).map(c => TERRAINS[c]), tokens: o.slice(147, 166), robber: o[220],
        harbors: o.slice(166, 220).map(h => h === 0 ? null : h === 1 ? "3:1" : RESOURCES[h - 2]), settlements, cities, roads,
      },
      seats, hand: o.slice(222, 227), devCards: o.slice(258, 263), bought: o.slice(263, 268), rates: o.slice(RATES_AT, RATES_AT + 5),
      bank: o.slice(253, 258), deck: o[274], phase: PHASES[o[296]], rolled: !!o[297], count: o[298], current: seatOf(o[294]),
      phasePlayer: o[299] === 255 ? null : seatOf(o[299]), cardPlayed: !!o[295],
      trade: {
        offer: offer.some(Boolean) ? { give: offer.slice(0, 5), get: offer.slice(5) } : null,
        draft: { give, get }, drafting: ["", "offer", "counter"][drafting], offers: o[OFFERS_MADE], budgetLeft: o[BUDGET_LEFT], closed,
        last: last.some(Boolean) ? { give: last.slice(0, 5), get: last.slice(5), traded: !!o[LAST_TRADED] } : null,
      },
      lastRoll: this.lastRoll, moves: this.moves(), events: this.events.slice(-60), payouts: this.payouts.slice(-4), insight: e.winner() == null ? this.insight() : null,
      forbidden: e.toAct() === this.human ? e.forbidden() : [], offerBoard: trade,
    };
  }
}
