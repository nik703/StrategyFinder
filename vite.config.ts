import type { IncomingMessage, ServerResponse } from 'node:http';
import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import { defineConfig, type Plugin } from 'vite';

const FUNCTION_ROUTE = '/.netlify/functions/market-data';

/**
 * Serves the market-data Netlify Function straight from the Vite dev server, so
 * plain `npm run dev` gets live data too. Under `netlify dev` the Netlify proxy
 * answers `/.netlify/functions/*` itself and this middleware is never reached.
 */
function netlifyFunctionDevServer(): Plugin {
  return {
    name: 'strategy-finder:netlify-function-dev',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use(FUNCTION_ROUTE, (req: IncomingMessage, res: ServerResponse) => {
        void (async () => {
          try {
            const mod = (await server.ssrLoadModule('/netlify/functions/market-data.ts')) as {
              default: (request: Request) => Promise<Response>;
            };
            const url = new URL(req.originalUrl ?? req.url ?? '/', 'http://localhost');
            const response = await mod.default(new Request(url, { method: req.method ?? 'GET' }));
            res.statusCode = response.status;
            response.headers.forEach((value, key) => res.setHeader(key, value));
            res.end(Buffer.from(await response.arrayBuffer()));
          } catch (error) {
            res.statusCode = 500;
            res.setHeader('Content-Type', 'application/json; charset=utf-8');
            res.end(JSON.stringify({ error: error instanceof Error ? error.message : 'Function crashed' }));
          }
        })();
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), tailwindcss(), netlifyFunctionDevServer()],
  server: {
    port: 5173,
    strictPort: true,
  },
  preview: {
    port: 4173,
  },
  worker: {
    format: 'es',
  },
  build: {
    target: 'es2022',
    sourcemap: false,
  },
});
