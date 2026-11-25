import '@fontsource-variable/inter';
import './index.css';

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { initStorage } from './lib/storage';

const container = document.getElementById('root');
if (!container) throw new Error('Root element #root is missing from index.html');

// The trial ledger, holdout log and saved runs are read synchronously by the app, so IndexedDB is
// loaded (and legacy localStorage migrated) before the first render. It never rejects.
void initStorage().then(() => {
  createRoot(container).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
});
