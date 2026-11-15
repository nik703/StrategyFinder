import { beforeEach, describe, expect, it } from 'vitest';
import { addSavedTickers, loadSavedTickers, MAX_SAVED_TICKERS, removeSavedTicker } from '../src/lib/savedTickers';

/** In-memory stand-in for window.localStorage. */
class MemoryStorage {
  private data = new Map<string, string>();
  getItem(key: string): string | null {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.data.set(key, value);
  }
  clear(): void {
    this.data.clear();
  }
}

const storage = new MemoryStorage();
(globalThis as unknown as { window: { localStorage: MemoryStorage } }).window = { localStorage: storage };

beforeEach(() => storage.clear());

describe('saved tickers', () => {
  it('remembers custom tickers across page loads, in the order they were first saved', () => {
    let list = addSavedTickers(loadSavedTickers(), ['tsla']);
    list = addSavedTickers(list, ['MSFT', 'TSLA']);
    expect(list).toEqual(['TSLA', 'MSFT']);
    // A fresh page load reads the same list back.
    expect(loadSavedTickers()).toEqual(['TSLA', 'MSFT']);
  });

  it('ignores presets, invalid symbols and duplicates', () => {
    const list = addSavedTickers([], ['BTC-USD', 'spy', 'bad ticker!', '', 'DOGE-USD', 'doge-usd']);
    expect(list).toEqual(['DOGE-USD']);
  });

  it('keeps the most recent entries when the list is full, and can forget one', () => {
    let list: string[] = [];
    for (let i = 0; i < MAX_SAVED_TICKERS + 3; i++) list = addSavedTickers(list, [`T${i}`]);
    expect(list).toHaveLength(MAX_SAVED_TICKERS);
    expect(list[0]).toBe('T3');
    expect(list[list.length - 1]).toBe(`T${MAX_SAVED_TICKERS + 2}`);
    list = removeSavedTicker(list, 'T5');
    expect(list).not.toContain('T5');
    expect(loadSavedTickers()).toEqual(list);
  });

  it('survives corrupt or unavailable storage', () => {
    storage.setItem('strategyfinder.savedTickers.v1', '{not json');
    expect(loadSavedTickers()).toEqual([]);
    storage.setItem('strategyfinder.savedTickers.v1', JSON.stringify(['NFLX', 42, null, 'nflx', 'AAPL']));
    expect(loadSavedTickers()).toEqual(['NFLX']);
    const broken = { getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } };
    (globalThis as unknown as { window: { localStorage: unknown } }).window = { localStorage: broken };
    expect(loadSavedTickers()).toEqual([]);
    expect(addSavedTickers([], ['AMD'])).toEqual(['AMD']); // still works for the session
    (globalThis as unknown as { window: { localStorage: MemoryStorage } }).window = { localStorage: storage };
  });
});
