/**
 * Persistent storage: one typed IndexedDB database with three object stores.
 *
 *   candles  incremental historical candle cache, one record per (exchange, symbol, timeframe):
 *            only closed bars are stored, and later runs download just the candles after the last
 *            stored one (see `data/candleCache.ts`);
 *   ledger   multiple-testing state: the cumulative trial counts of every data window, the budget
 *            extensions granted on them and every locked-holdout reveal (see `trialLedger.ts` and
 *            `holdoutLog.ts`);
 *   runs     saved runs and their final leaderboard snapshots.
 *
 * The ledger and runs stores are small and read synchronously all over the app (the trial-budget
 * meter, the deflated Sharpe's hurdle, the holdout warning), so they are mirrored in memory: `init`
 * loads them once at start-up, reads come from the mirror, and every write updates the mirror at
 * once and is persisted to IndexedDB in the background (`flushWrites` waits for it). The candle
 * store can be large and is only read asynchronously.
 *
 * On first load the legacy localStorage keys of the trial ledger and holdout log are migrated into
 * the ledger store and removed. Without IndexedDB (private mode, very old browsers, Node tests) the
 * mirror still works for the session and nothing is persisted.
 *
 * `exportSnapshot` / `importSnapshot` move the ledger and the saved runs between browsers as one
 * JSON document (candles are not included: they are re-downloaded).
 */

export const DB_NAME = 'strategyfinder';
export const DB_VERSION = 1;
export const STORES = { candles: 'candles', ledger: 'ledger', runs: 'runs' } as const;
export type StoreName = (typeof STORES)[keyof typeof STORES];

/** Kinds of record in the ledger store. */
export type LedgerKind = 'trials' | 'extension' | 'holdout';

/** One ledger record: a kind, a unique id and the payload of `trialLedger.ts` / `holdoutLog.ts`. */
export interface LedgerRecord<T = unknown> {
  id: string;
  kind: LedgerKind;
  data: T;
}

/** A saved run with its final leaderboard (JSON-safe: equity curves as plain number arrays). */
export interface SavedRun {
  id: string;
  savedAt: number;
  label: string;
  interval: string;
  range: string;
  hypothesis: string;
  tradeMode: string;
  /** The strategy whose holdout was revealed, if any. */
  lockedId: string | null;
  strategies: unknown[];
}

/** Closed candles of one (exchange, symbol, timeframe). */
export interface CandleRecord {
  /** `exchange|SYMBOL|interval`. */
  key: string;
  exchange: string;
  symbol: string;
  interval: string;
  venueSymbol?: string;
  t: number[];
  o: number[];
  h: number[];
  l: number[];
  c: number[];
  v: number[];
  updatedAt: number;
}

/* -------------------------------------------------------------------------- */
/*  IndexedDB plumbing                                                        */
/* -------------------------------------------------------------------------- */

/** Wrap an IDBRequest in a Promise. */
function done<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed'));
  });
}

/** Wrap a transaction's completion in a Promise. */
function complete(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error('IndexedDB transaction failed'));
    tx.onabort = () => reject(tx.error ?? new Error('IndexedDB transaction aborted'));
  });
}

/** Open (and create or upgrade) the database. */
export function openDatabase(factory: IDBFactory): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = factory.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORES.candles)) db.createObjectStore(STORES.candles, { keyPath: 'key' });
      if (!db.objectStoreNames.contains(STORES.ledger)) db.createObjectStore(STORES.ledger, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(STORES.runs)) db.createObjectStore(STORES.runs, { keyPath: 'id' });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('Could not open IndexedDB'));
    request.onblocked = () => reject(new Error('IndexedDB upgrade blocked by another tab'));
  });
}

/* -------------------------------------------------------------------------- */
/*  State                                                                     */
/* -------------------------------------------------------------------------- */

let db: IDBDatabase | null = null;
let ready: Promise<StorageStatus> | null = null;
const ledger = new Map<string, LedgerRecord>();
const runs = new Map<string, SavedRun>();
/** Background writes still in flight. */
const writes = new Set<Promise<void>>();

export interface StorageStatus {
  /** True when IndexedDB is available and open (false: in-memory only, nothing persists). */
  persistent: boolean;
  /** Legacy localStorage records migrated into the ledger store on this load. */
  migrated: number;
}

/** The IndexedDB factory of this environment, or null. */
function defaultFactory(): IDBFactory | null {
  try {
    return typeof indexedDB !== 'undefined' ? indexedDB : null;
  } catch {
    return null;
  }
}

/**
 * Open the database, load the ledger and runs into memory and migrate legacy localStorage keys.
 * Safe to call more than once (later calls return the first call's result). Never rejects.
 */
export function initStorage(factory: IDBFactory | null = defaultFactory()): Promise<StorageStatus> {
  ready ??= (async (): Promise<StorageStatus> => {
    if (factory) {
      try {
        db = await openDatabase(factory);
        const tx = db.transaction([STORES.ledger, STORES.runs], 'readonly');
        const [ledgerRows, runRows] = await Promise.all([
          done(tx.objectStore(STORES.ledger).getAll() as IDBRequest<LedgerRecord[]>),
          done(tx.objectStore(STORES.runs).getAll() as IDBRequest<SavedRun[]>),
        ]);
        for (const row of ledgerRows) if (isLedgerRecord(row)) ledger.set(row.id, row);
        for (const row of runRows) if (isSavedRun(row)) runs.set(row.id, row);
      } catch {
        db = null; // blocked or broken: carry on in memory
      }
    }
    const migrated = migrateLegacyLocalStorage();
    if (migrated > 0) await flushWrites();
    return { persistent: db !== null, migrated };
  })();
  return ready;
}

/** Wait until every background write has reached IndexedDB. */
export async function flushWrites(): Promise<void> {
  while (writes.size > 0) await Promise.allSettled([...writes]);
}

function track(write: Promise<void>): void {
  const settled = write.catch(() => undefined).finally(() => writes.delete(settled));
  writes.add(settled);
}

/** Put / delete records in one store in the background. */
function persist(store: StoreName, put: readonly unknown[], remove: readonly string[] = [], clear = false): void {
  const database = db;
  if (!database) return;
  track(
    (async () => {
      const tx = database.transaction(store, 'readwrite');
      const os = tx.objectStore(store);
      if (clear) os.clear();
      for (const key of remove) os.delete(key);
      for (const value of put) os.put(value);
      await complete(tx);
    })(),
  );
}

/* -------------------------------------------------------------------------- */
/*  Ledger (synchronous mirror)                                               */
/* -------------------------------------------------------------------------- */

function isLedgerRecord(value: unknown): value is LedgerRecord {
  if (typeof value !== 'object' || value === null) return false;
  const r = value as Record<string, unknown>;
  return typeof r.id === 'string' && (r.kind === 'trials' || r.kind === 'extension' || r.kind === 'holdout') && typeof r.data === 'object' && r.data !== null;
}

/** Every ledger record of `kind`, as stored (callers validate the payload). */
export function ledgerRecords<T>(kind: LedgerKind): LedgerRecord<T>[] {
  const out: LedgerRecord<T>[] = [];
  for (const record of ledger.values()) if (record.kind === kind) out.push(record as LedgerRecord<T>);
  return out;
}

/** Insert or replace ledger records (mirror now, IndexedDB in the background). */
export function putLedgerRecords(records: readonly LedgerRecord[]): void {
  for (const record of records) ledger.set(record.id, record);
  persist(STORES.ledger, records);
}

/** Delete ledger records by id. */
export function deleteLedgerRecords(ids: readonly string[]): void {
  for (const id of ids) ledger.delete(id);
  persist(STORES.ledger, [], ids);
}

/* -------------------------------------------------------------------------- */
/*  Saved runs                                                                */
/* -------------------------------------------------------------------------- */

function isSavedRun(value: unknown): value is SavedRun {
  if (typeof value !== 'object' || value === null) return false;
  const r = value as Record<string, unknown>;
  return typeof r.id === 'string' && typeof r.savedAt === 'number' && Array.isArray(r.strategies);
}

/** Oldest saved runs are dropped beyond this many. */
export const MAX_SAVED_RUNS = 50;

/** Saved runs, newest first. */
export function savedRuns(): SavedRun[] {
  return [...runs.values()].sort((a, b) => b.savedAt - a.savedAt);
}

/** Save (or update) a run and its leaderboard snapshot. Typed arrays are converted to plain arrays. */
export function saveRun(run: SavedRun): void {
  const clean = jsonSafe(run) as SavedRun;
  runs.set(clean.id, clean);
  const drop = savedRuns().slice(MAX_SAVED_RUNS).map((r) => r.id);
  for (const id of drop) runs.delete(id);
  persist(STORES.runs, [clean], drop);
}

/** Deep copy with typed arrays turned into number arrays and non-finite numbers into null. */
export function jsonSafe(value: unknown): unknown {
  return JSON.parse(
    JSON.stringify(value, (_key, v: unknown) => {
      if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v as unknown as ArrayLike<number>);
      return v;
    }),
  );
}

/* -------------------------------------------------------------------------- */
/*  Candles (asynchronous)                                                    */
/* -------------------------------------------------------------------------- */

export const candleKey = (exchange: string, symbol: string, interval: string): string => `${exchange}|${symbol.toUpperCase()}|${interval}`;

/** Every cached candle record of `symbol` on `interval`, whatever the exchange. */
export async function readCandles(symbol: string, interval: string): Promise<CandleRecord[]> {
  await initStorage();
  const database = db;
  if (!database) return memoryCandles.filter((r) => r.symbol === symbol.toUpperCase() && r.interval === interval);
  try {
    const tx = database.transaction(STORES.candles, 'readonly');
    const rows = (await done(tx.objectStore(STORES.candles).getAll() as IDBRequest<CandleRecord[]>)) ?? [];
    return rows.filter((r) => r.symbol === symbol.toUpperCase() && r.interval === interval);
  } catch {
    return [];
  }
}

/** In-memory candle cache when IndexedDB is unavailable (one session only). */
const memoryCandles: CandleRecord[] = [];

/** Store closed candles of one (exchange, symbol, timeframe), replacing what was there. */
export async function writeCandles(record: CandleRecord): Promise<void> {
  await initStorage();
  const at = memoryCandles.findIndex((r) => r.key === record.key);
  if (!db) {
    if (at >= 0) memoryCandles[at] = record;
    else memoryCandles.push(record);
    return;
  }
  persist(STORES.candles, [record]);
  await flushWrites();
}

/* -------------------------------------------------------------------------- */
/*  Reset, snapshots, legacy migration                                        */
/* -------------------------------------------------------------------------- */

/**
 * "Reset Runs & Clear Cache": empty the ledger (trials, budget extensions, holdout reveals), the
 * saved runs and the candle cache. The mirror is cleared at once; IndexedDB in the background.
 */
export function clearAllRunData(): void {
  ledger.clear();
  runs.clear();
  memoryCandles.length = 0;
  persist(STORES.ledger, [], [], true);
  persist(STORES.runs, [], [], true);
  persist(STORES.candles, [], [], true);
}

export const SNAPSHOT_FORMAT = 'strategyfinder-snapshot';
export const SNAPSHOT_VERSION = 1;

export interface Snapshot {
  format: typeof SNAPSHOT_FORMAT;
  version: number;
  exportedAt: number;
  ledger: LedgerRecord[];
  runs: SavedRun[];
}

/** Everything worth backing up: the multiple-testing ledger and the saved runs. */
export function exportSnapshot(now: number = Date.now()): Snapshot {
  return {
    format: SNAPSHOT_FORMAT,
    version: SNAPSHOT_VERSION,
    exportedAt: now,
    ledger: [...ledger.values()].map((record) => jsonSafe(record) as LedgerRecord),
    runs: savedRuns(),
  };
}

export interface ImportSummary {
  trials: number;
  extensions: number;
  holdouts: number;
  runs: number;
  /** Records that were malformed and skipped. */
  skipped: number;
}

/**
 * Merge a snapshot into this browser's storage. Records are keyed by id, so importing the same
 * snapshot twice changes nothing and nothing already recorded here is lost: an import can only add
 * trials and holdout reveals, never erase multiple-testing history. Throws on a foreign file.
 */
export function importSnapshot(input: unknown): ImportSummary {
  const parsed = typeof input === 'string' ? (JSON.parse(input) as unknown) : input;
  if (typeof parsed !== 'object' || parsed === null || (parsed as Snapshot).format !== SNAPSHOT_FORMAT) {
    throw new Error('Not a StrategyFinder snapshot file.');
  }
  const snapshot = parsed as Partial<Snapshot>;
  if (typeof snapshot.version !== 'number' || snapshot.version > SNAPSHOT_VERSION) {
    throw new Error(`Unsupported snapshot version ${String(snapshot.version)}.`);
  }
  const summary: ImportSummary = { trials: 0, extensions: 0, holdouts: 0, runs: 0, skipped: 0 };
  const records: LedgerRecord[] = [];
  for (const record of Array.isArray(snapshot.ledger) ? snapshot.ledger : []) {
    if (!isLedgerRecord(record)) {
      summary.skipped++;
      continue;
    }
    // Trial records of the same run are replaced by the larger count; everything else is added.
    const existing = ledger.get(record.id);
    if (record.kind === 'trials' && existing && countOf(existing) > countOf(record)) continue;
    records.push(record);
    if (record.kind === 'trials') summary.trials++;
    else if (record.kind === 'extension') summary.extensions++;
    else summary.holdouts++;
  }
  putLedgerRecords(records);
  const imported: SavedRun[] = [];
  for (const run of Array.isArray(snapshot.runs) ? snapshot.runs : []) {
    if (!isSavedRun(run)) {
      summary.skipped++;
      continue;
    }
    imported.push(run);
  }
  for (const run of imported) runs.set(run.id, run);
  persist(STORES.runs, imported);
  summary.runs = imported.length;
  return summary;
}

const countOf = (record: LedgerRecord): number => {
  const count = (record.data as { count?: unknown }).count;
  return typeof count === 'number' ? count : 0;
};

/** localStorage keys written by earlier versions; migrated into the ledger store and removed. */
export const LEGACY_KEYS = {
  trials: 'strategyfinder.trialLedger.v1',
  extensions: 'strategyfinder.budgetExtensions.v1',
  holdouts: 'strategyfinder.holdoutReveals.v2',
  holdoutsV1: 'strategyfinder.holdoutViews.v1',
} as const;

/** Legacy migration hooks registered by the ledger modules (they know their own record formats). */
const migrators: ((storage: Storage) => LedgerRecord[])[] = [];

/** Register a function that turns a module's legacy localStorage keys into ledger records. */
export function registerLegacyMigration(migrate: (storage: Storage) => LedgerRecord[]): void {
  migrators.push(migrate);
}

function browserLocalStorage(): Storage | null {
  try {
    return typeof window !== 'undefined' && window.localStorage ? window.localStorage : null;
  } catch {
    return null;
  }
}

/** Run every registered migration once, then remove the legacy keys. Returns the records migrated. */
function migrateLegacyLocalStorage(): number {
  const storage = browserLocalStorage();
  if (!storage) return 0;
  let count = 0;
  for (const migrate of migrators) {
    try {
      const records = migrate(storage).filter((record) => !ledger.has(record.id));
      putLedgerRecords(records);
      count += records.length;
    } catch {
      // A corrupt legacy key is dropped below.
    }
  }
  for (const key of Object.values(LEGACY_KEYS)) {
    try {
      storage.removeItem(key);
    } catch {
      // Storage blocked: nothing to remove.
    }
  }
  return count;
}

/** Test helper: forget the open database and the mirror, so `initStorage` runs again. */
export function resetStorageForTests(): void {
  db?.close();
  db = null;
  ready = null;
  ledger.clear();
  runs.clear();
  memoryCandles.length = 0;
  writes.clear();
}
