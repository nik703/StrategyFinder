/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional absolute URL of a deployed market-data function. */
  readonly VITE_MARKET_DATA_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
