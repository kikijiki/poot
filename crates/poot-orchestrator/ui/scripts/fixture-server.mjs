// Serves the built dashboard page and the fixture API (src/fixtures.ts) on a loopback port, so the screenshots show
// the same states on every run. `?__theme=light|dark` makes the page start in that theme.
import {createServer} from 'node:http';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';

const bundled = await build({
  entryPoints: ['src/fixtures.ts'],
  bundle: true,
  write: false,
  platform: 'node',
  format: 'esm',
  loader: {'.css': 'text'},
  define: {__NOW__: String(Date.now())},
});
const fx = await import(`data:text/javascript;base64,${Buffer.from(bundled.outputFiles[0].text).toString('base64')}`);

export function startFixtureServer(port, htmlPath = 'dist/index.html') {
  const page = readFileSync(htmlPath, 'utf8');
  const json = (res, body) => {
    res.writeHead(200, {'content-type': 'application/json'});
    res.end(JSON.stringify(body));
  };
  const server = createServer((req, res) => {
    const url = new URL(req.url, 'http://x');
    switch (url.pathname) {
      case '/':
      case '/index.html': {
        const theme = url.searchParams.get('__theme');
        const boot = theme ? `<script>localStorage.setItem('poot-theme','${theme}')</script>` : '<script>localStorage.removeItem("poot-theme")</script>';
        res.writeHead(200, {'content-type': 'text/html; charset=utf-8'});
        res.end(page.replace('<head>', `<head>${boot}`));
        return;
      }
      case '/api/state':
        return json(res, fx.STATE);
      case '/api/overview':
        return json(res, fx.OVERVIEW);
      case '/api/logfiles':
        return json(res, fx.LOGFILES);
      case '/api/results':
        return json(res, fx.resultsOf(url.searchParams.get('run') ?? ''));
      case '/api/log':
        res.writeHead(200, {'content-type': 'text/plain; charset=utf-8'});
        return res.end(fx.logOf(url.searchParams.get('name') ?? ''));
      case '/api/serve-metrics':
        return json(res, {error: 'connect 127.0.0.1:8080: connection refused'});
      default:
        res.writeHead(404);
        return res.end('not found');
    }
  });
  return new Promise((resolve) => server.listen(port, '127.0.0.1', () => resolve(server)));
}
