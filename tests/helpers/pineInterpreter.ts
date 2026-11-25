/**
 * A tiny interpreter for the subset of Pine Script v5 that StrategyFinder's exporter emits – used by
 * the parity tests to run a GENERATED script on the same candles as the TypeScript engine.
 *
 *   • Definitions (`name = expr`, `[a, b, c] = expr`, `input.*`) are evaluated as whole series, on
 *     demand. `ta.*` built-ins map to the engine's indicator library (which mirrors TradingView's
 *     formulas), so a parity failure points at the TRANSLATION – the operands, operators, offsets,
 *     mirroring and direction gating the exporter wrote – not at TradingView's numerics.
 *   • The exit-management block (`var` state, nested `if` / `else`, `:=`, `strategy.exit(...)`) is
 *     executed bar by bar against the positions the engine actually held, and records the stop and
 *     limit each `strategy.exit` call sends.
 *
 * Semantics follow Pine: `na` comparisons are false, `x[n]` looks n candles back, `and` / `or`
 * treat na as false.
 */

import { adx, atr, bollinger, cci, ema, macd, rsi, sma, stochastic, supertrend, trueRange } from '../../src/engine/indicators';
import type { OHLCV } from '../../src/types/trading';

/* -------------------------------------------------------------------------- */
/*  Lexer and parser                                                          */
/* -------------------------------------------------------------------------- */

type Tok = { t: 'num'; v: number } | { t: 'id'; v: string } | { t: 'str'; v: string } | { t: 'op'; v: string };

function lex(src: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < src.length) {
    const ch = src[i];
    if (ch === ' ' || ch === '\t') {
      i++;
      continue;
    }
    if (ch === '/' && src[i + 1] === '/') break; // comment
    if (/[0-9]/.test(ch) || (ch === '.' && /[0-9]/.test(src[i + 1] ?? ''))) {
      let j = i;
      while (j < src.length && /[0-9.eE]/.test(src[j])) j++;
      out.push({ t: 'num', v: Number(src.slice(i, j)) });
      i = j;
      continue;
    }
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < src.length && /[A-Za-z0-9_.]/.test(src[j])) j++;
      out.push({ t: 'id', v: src.slice(i, j) });
      i = j;
      continue;
    }
    if (ch === '"') {
      const j = src.indexOf('"', i + 1);
      out.push({ t: 'str', v: src.slice(i + 1, j) });
      i = j + 1;
      continue;
    }
    const two = src.slice(i, i + 2);
    if (['>=', '<=', '==', '!=', ':='].includes(two)) {
      out.push({ t: 'op', v: two });
      i += 2;
      continue;
    }
    out.push({ t: 'op', v: ch });
    i++;
  }
  return out;
}

export type Expr =
  | { e: 'num'; v: number }
  | { e: 'str'; v: string }
  | { e: 'id'; v: string }
  | { e: 'call'; fn: string; args: Expr[]; named: Record<string, Expr> }
  | { e: 'index'; of: Expr; by: Expr }
  | { e: 'un'; op: string; a: Expr }
  | { e: 'bin'; op: string; a: Expr; b: Expr }
  | { e: 'tern'; c: Expr; a: Expr; b: Expr };

class Parser {
  private k = 0;
  constructor(private readonly toks: Tok[]) {}
  private peek(): Tok | undefined {
    return this.toks[this.k];
  }
  private isOp(v: string): boolean {
    const t = this.peek();
    return t !== undefined && ((t.t === 'op' && t.v === v) || (t.t === 'id' && t.v === v));
  }
  private take(): Tok {
    return this.toks[this.k++];
  }
  private expect(v: string): void {
    if (!this.isOp(v)) throw new Error(`Pine parse: expected ${v} at ${JSON.stringify(this.peek())}`);
    this.k++;
  }
  done(): boolean {
    return this.k >= this.toks.length;
  }
  expr(): Expr {
    const c = this.or();
    if (this.isOp('?')) {
      this.take();
      const a = this.expr();
      this.expect(':');
      const b = this.expr();
      return { e: 'tern', c, a, b };
    }
    return c;
  }
  private or(): Expr {
    let a = this.and();
    while (this.isOp('or')) {
      this.take();
      a = { e: 'bin', op: 'or', a, b: this.and() };
    }
    return a;
  }
  private and(): Expr {
    let a = this.not();
    while (this.isOp('and')) {
      this.take();
      a = { e: 'bin', op: 'and', a, b: this.not() };
    }
    return a;
  }
  private not(): Expr {
    if (this.isOp('not')) {
      this.take();
      return { e: 'un', op: 'not', a: this.not() };
    }
    return this.cmp();
  }
  private cmp(): Expr {
    let a = this.add();
    for (;;) {
      const t = this.peek();
      if (t?.t === 'op' && ['>', '<', '>=', '<=', '==', '!='].includes(t.v)) {
        this.take();
        a = { e: 'bin', op: t.v, a, b: this.add() };
      } else return a;
    }
  }
  private add(): Expr {
    let a = this.mul();
    for (;;) {
      const t = this.peek();
      if (t?.t === 'op' && (t.v === '+' || t.v === '-')) {
        this.take();
        a = { e: 'bin', op: t.v, a, b: this.mul() };
      } else return a;
    }
  }
  private mul(): Expr {
    let a = this.unary();
    for (;;) {
      const t = this.peek();
      if (t?.t === 'op' && (t.v === '*' || t.v === '/')) {
        this.take();
        a = { e: 'bin', op: t.v, a, b: this.unary() };
      } else return a;
    }
  }
  private unary(): Expr {
    if (this.isOp('-')) {
      this.take();
      return { e: 'un', op: '-', a: this.unary() };
    }
    return this.postfix();
  }
  private postfix(): Expr {
    let a = this.primary();
    for (;;) {
      if (this.isOp('(') && a.e === 'id') {
        this.take();
        const args: Expr[] = [];
        const named: Record<string, Expr> = {};
        while (!this.isOp(')')) {
          const t = this.peek();
          const next = this.toks[this.k + 1];
          if (t?.t === 'id' && next?.t === 'op' && next.v === '=') {
            this.k += 2;
            named[t.v] = this.expr();
          } else args.push(this.expr());
          if (this.isOp(',')) this.take();
        }
        this.expect(')');
        a = { e: 'call', fn: a.v, args, named };
      } else if (this.isOp('[')) {
        this.take();
        const by = this.expr();
        this.expect(']');
        a = { e: 'index', of: a, by };
      } else return a;
    }
  }
  private primary(): Expr {
    const t = this.take();
    if (!t) throw new Error('Pine parse: unexpected end');
    if (t.t === 'num') return { e: 'num', v: t.v };
    if (t.t === 'str') return { e: 'str', v: t.v };
    if (t.t === 'id') return { e: 'id', v: t.v };
    if (t.v === '(') {
      const e = this.expr();
      this.expect(')');
      return e;
    }
    throw new Error(`Pine parse: unexpected ${t.v}`);
  }
}

export function parseExpr(src: string): Expr {
  const p = new Parser(lex(src));
  const e = p.expr();
  if (!p.done()) throw new Error(`Pine parse: trailing tokens in "${src}"`);
  return e;
}

/* -------------------------------------------------------------------------- */
/*  Interpreter                                                               */
/* -------------------------------------------------------------------------- */

type Series = Float64Array;
const truthy = (v: number): boolean => v === v && v !== 0;

interface Stmt {
  indent: number;
  text: string;
  children: Stmt[];
}

/** Recorded `strategy.exit` stop / limit per candle, by order id. */
export type ExitOrders = Record<string, { stop: Float64Array; limit: Float64Array }>;

export class PineProgram {
  private readonly defs = new Map<string, string>();
  private readonly tuples = new Map<string, { names: string[]; expr: string }>();
  private readonly cache = new Map<string, Series | Series[]>();
  readonly n: number;
  private readonly exitBlock: Stmt[];

  constructor(
    script: string,
    private readonly data: OHLCV,
    /** Extra series (e.g. `strategy.position_size`) the script may read. */
    private readonly env: Record<string, Series> = {},
  ) {
    this.n = data.c.length;
    const lines = script.split('\n');
    const start = lines.findIndex((l) => l.startsWith('// ---- Exit management'));
    const stop = lines.findIndex((l, i) => i > start && /^(\/\/ (Opposite|Flatten)|if timedOut|if longExitSignal|\/\/ ---- Plots)/.test(l));
    const exitLines = start >= 0 ? lines.slice(start + 1, stop > start ? stop : undefined) : [];
    lines.forEach((line, i) => {
      if (start >= 0 && i > start && (stop < 0 || i < stop)) return;
      if (/^\s/.test(line) || line.startsWith('//') || line.startsWith('if ') || line.startsWith('var ')) return;
      const tuple = /^\[([^\]]+)\]\s*=\s*(.+)$/.exec(line);
      if (tuple) {
        const names = tuple[1].split(',').map((x) => x.trim());
        for (const name of names) this.tuples.set(name, { names, expr: tuple[2] });
        return;
      }
      const def = /^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$/.exec(line);
      if (def) this.defs.set(def[1], def[2]);
    });
    this.exitBlock = this.block(exitLines.filter((l) => l.trim() !== '' && !l.trim().startsWith('//')));
  }

  /** The whole series of a defined name (a script variable or a built-in). */
  series(name: string): Series {
    return this.evalSeries({ e: 'id', v: name });
  }

  private builtin(name: string): Series | null {
    const { o, h, l, c, v } = this.data;
    switch (name) {
      case 'close':
        return c;
      case 'open':
        return o;
      case 'high':
        return h;
      case 'low':
        return l;
      case 'volume':
        return v;
      case 'hlc3': {
        const out = new Float64Array(this.n);
        for (let i = 0; i < this.n; i++) out[i] = (h[i] + l[i] + c[i]) / 3;
        return out;
      }
      case 'na':
        return new Float64Array(this.n).fill(NaN);
      case 'true':
        return new Float64Array(this.n).fill(1);
      case 'false':
        return new Float64Array(this.n).fill(0);
    }
    return this.env[name] ?? null;
  }

  private fill(value: number): Series {
    return new Float64Array(this.n).fill(value);
  }

  private map2(a: Series, b: Series, f: (x: number, y: number) => number): Series {
    const out = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) out[i] = f(a[i], b[i]);
    return out;
  }

  private shift(a: Series, k: number): Series {
    const out = new Float64Array(this.n).fill(NaN);
    for (let i = k; i < this.n; i++) out[i] = a[i - k];
    return out;
  }

  private window(src: Series, len: number, pick: (a: number, b: number) => number): Series {
    const out = new Float64Array(this.n).fill(NaN);
    for (let i = len - 1; i < this.n; i++) {
      let best = src[i];
      for (let j = i - len + 1; j < i; j++) best = pick(best, src[j]);
      out[i] = best;
    }
    return out;
  }

  private call(fn: string, args: Expr[]): Series | Series[] {
    const S = (k: number): Series => this.evalSeries(args[k]);
    const num = (k: number): number => {
      const s = this.evalSeries(args[k]);
      return s[s.length - 1];
    };
    const { h, l, c } = this.data;
    switch (fn) {
      case 'input.float':
      case 'input.int':
      case 'input.bool':
      case 'input.time':
        return S(0);
      case 'ta.ema':
        return ema(S(0), num(1));
      case 'ta.sma':
        return sma(S(0), num(1));
      case 'ta.rsi':
        return rsi(S(0), num(1));
      case 'ta.atr':
        return atr(h, l, c, num(0));
      case 'ta.tr':
        return trueRange(h, l, c);
      case 'ta.bb': {
        const r = bollinger(S(0), num(1), num(2));
        return [r.middle, r.upper, r.lower];
      }
      case 'ta.macd': {
        const r = macd(S(0), num(1), num(2), num(3));
        return [r.line, r.signal, r.hist];
      }
      case 'ta.dmi': {
        const r = adx(h, l, c, num(0));
        return [r.plusDI, r.minusDI, r.adx];
      }
      case 'ta.stoch':
        return stochastic(h, l, c, num(3), 1, 1).k;
      case 'ta.cci':
        return cci(h, l, c, num(1));
      case 'ta.supertrend': {
        const r = supertrend(h, l, c, num(1), num(0));
        return [r.line, r.direction];
      }
      case 'ta.highest':
        return this.window(S(0), num(1), Math.max);
      case 'ta.lowest':
        return this.window(S(0), num(1), Math.min);
      case 'ta.crossover':
      case 'ta.crossunder': {
        const a = S(0);
        const b = S(1);
        const out = new Float64Array(this.n);
        for (let i = 1; i < this.n; i++) {
          out[i] = fn === 'ta.crossover' ? (a[i] > b[i] && a[i - 1] <= b[i - 1] ? 1 : 0) : a[i] < b[i] && a[i - 1] >= b[i - 1] ? 1 : 0;
        }
        return out;
      }
      case 'na': {
        const a = S(0);
        return a.map((x) => (x !== x ? 1 : 0));
      }
      case 'math.max':
        return this.map2(S(0), S(1), Math.max);
      case 'math.min':
        return this.map2(S(0), S(1), Math.min);
    }
    throw new Error(`Pine: unsupported function ${fn}`);
  }

  private evalSeries(e: Expr): Series {
    const key = JSON.stringify(e);
    const hit = this.cache.get(key);
    if (hit) return hit as Series;
    const value = this.compute(e);
    this.cache.set(key, value);
    return value;
  }

  private compute(e: Expr): Series {
    switch (e.e) {
      case 'num':
        return this.fill(e.v);
      case 'str':
        return this.fill(NaN);
      case 'id': {
        const builtin = this.builtin(e.v);
        if (builtin) return builtin;
        const def = this.defs.get(e.v);
        if (def !== undefined) return this.evalSeries(parseExpr(def));
        const tuple = this.tuples.get(e.v);
        if (tuple) {
          const call = parseExpr(tuple.expr);
          if (call.e !== 'call') throw new Error('Pine: tuple without a call');
          const parts = this.call(call.fn, call.args) as Series[];
          return parts[tuple.names.indexOf(e.v)];
        }
        throw new Error(`Pine: unknown identifier ${e.v}`);
      }
      case 'call':
        return this.call(e.fn, e.args) as Series;
      case 'index': {
        const k = this.evalSeries(e.by)[0];
        return this.shift(this.evalSeries(e.of), k);
      }
      case 'un': {
        const a = this.evalSeries(e.a);
        return e.op === '-' ? a.map((x) => -x) : a.map((x) => (truthy(x) ? 0 : 1));
      }
      case 'bin': {
        const a = this.evalSeries(e.a);
        const b = this.evalSeries(e.b);
        return this.map2(a, b, (x, y) => binary(e.op, x, y));
      }
      case 'tern': {
        const c = this.evalSeries(e.c);
        const a = this.evalSeries(e.a);
        const b = this.evalSeries(e.b);
        const out = new Float64Array(this.n);
        for (let i = 0; i < this.n; i++) out[i] = truthy(c[i]) ? a[i] : b[i];
        return out;
      }
    }
  }

  /* ---- Statements (exit management) ------------------------------------- */

  private block(lines: string[]): Stmt[] {
    const root: Stmt = { indent: -1, text: '', children: [] };
    const stack: Stmt[] = [root];
    for (const line of lines) {
      const indent = line.length - line.trimStart().length;
      const node: Stmt = { indent, text: line.trim().replace(/\s*\/\/.*$/, ''), children: [] };
      while (stack[stack.length - 1].indent >= indent) stack.pop();
      stack[stack.length - 1].children.push(node);
      stack.push(node);
    }
    return root.children;
  }

  /** Scalar value of `e` on candle `i`, with `vars` holding the `var` state. */
  private at(e: Expr, i: number, vars: Map<string, number>): number {
    switch (e.e) {
      case 'id':
        if (vars.has(e.v)) return vars.get(e.v) as number;
        return i >= 0 ? this.evalSeries(e)[i] : NaN;
      case 'index': {
        const k = this.at(e.by, i, vars);
        if (e.of.e === 'id' && vars.has(e.of.v)) throw new Error('Pine: history of var state is not supported');
        return i - k >= 0 ? this.evalSeries(e.of)[i - k] : NaN;
      }
      case 'num':
        return e.v;
      case 'un': {
        const a = this.at(e.a, i, vars);
        return e.op === '-' ? -a : truthy(a) ? 0 : 1;
      }
      case 'bin':
        return binary(e.op, this.at(e.a, i, vars), this.at(e.b, i, vars));
      case 'tern':
        return truthy(this.at(e.c, i, vars)) ? this.at(e.a, i, vars) : this.at(e.b, i, vars);
      case 'call':
        if (e.fn === 'math.max' || e.fn === 'math.min') {
          const a = this.at(e.args[0], i, vars);
          const b = this.at(e.args[1], i, vars);
          // Pine's math.max/min return na when an argument is na.
          if (a !== a || b !== b) return NaN;
          return e.fn === 'math.max' ? Math.max(a, b) : Math.min(a, b);
        }
        if (e.fn === 'na') return this.at(e.args[0], i, vars) !== this.at(e.args[0], i, vars) ? 1 : 0;
        return this.evalSeries(e)[i];
      default:
        return NaN;
    }
  }

  /** Run the exit-management block on every candle and record each `strategy.exit` stop / limit. */
  runExitManagement(): ExitOrders {
    const vars = new Map<string, number>();
    const orders: ExitOrders = {};
    const exec = (stmts: Stmt[], i: number): void => {
      for (let k = 0; k < stmts.length; k++) {
        const s = stmts[k];
        if (s.text.startsWith('var ')) continue; // declared once, below
        if (s.text.startsWith('if ')) {
          const taken = truthy(this.at(parseExpr(s.text.slice(3)), i, vars));
          const next = stmts[k + 1];
          const hasElse = next !== undefined && next.text === 'else';
          if (taken) exec(s.children, i);
          else if (hasElse) exec(next.children, i);
          if (hasElse) k++;
          continue;
        }
        const assign = /^([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.+)$/.exec(s.text);
        if (assign) {
          vars.set(assign[1], this.at(parseExpr(assign[2]), i, vars));
          continue;
        }
        if (s.text.startsWith('strategy.exit(')) {
          const call = parseExpr(s.text);
          if (call.e !== 'call' || call.args[0]?.e !== 'str') throw new Error('Pine: malformed strategy.exit');
          const id = call.args[0].v;
          orders[id] ??= { stop: new Float64Array(this.n).fill(NaN), limit: new Float64Array(this.n).fill(NaN) };
          if (call.named.stop) orders[id].stop[i] = this.at(call.named.stop, i, vars);
          if (call.named.limit) orders[id].limit[i] = this.at(call.named.limit, i, vars);
          continue;
        }
        throw new Error(`Pine: unsupported statement "${s.text}"`);
      }
    };
    for (const s of this.exitBlock) {
      const decl = /^var float ([A-Za-z_][A-Za-z0-9_]*)\s*=\s*na$/.exec(s.text);
      if (decl) vars.set(decl[1], NaN);
    }
    for (let i = 0; i < this.n; i++) exec(this.exitBlock, i);
    return orders;
  }
}

function binary(op: string, x: number, y: number): number {
  switch (op) {
    case '+':
      return x + y;
    case '-':
      return x - y;
    case '*':
      return x * y;
    case '/':
      return x / y;
    case '>':
      return x > y ? 1 : 0;
    case '<':
      return x < y ? 1 : 0;
    case '>=':
      return x >= y ? 1 : 0;
    case '<=':
      return x <= y ? 1 : 0;
    case '==':
      return x === y ? 1 : 0;
    case '!=':
      return x !== y ? 1 : 0;
    case 'and':
      return truthy(x) && truthy(y) ? 1 : 0;
    case 'or':
      return truthy(x) || truthy(y) ? 1 : 0;
  }
  throw new Error(`Pine: unsupported operator ${op}`);
}
