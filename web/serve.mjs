// Serves web/dist at http://localhost:4331 with the headers from its _headers file, as Cloudflare Pages does, so the
// CSP is tested too. Only the `/*` block is read, which is all this site has.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';

const root = new URL('./dist/', import.meta.url).pathname;
const port = Number(process.env.PORT ?? 4331);
const types = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
};

const headers = Object.fromEntries(
  (await readFile(join(root, '_headers'), 'utf8'))
    .split('\n')
    .filter((line) => /^\s+\S+:/.test(line))
    .map((line) => line.trim().split(/:\s(.*)/s).slice(0, 2)),
);

createServer(async (request, response) => {
  let path = normalize(decodeURIComponent(new URL(request.url, 'http://x').pathname));
  if (path.endsWith('/')) path += 'index.html';
  try {
    const body = await readFile(join(root, path));
    response.writeHead(200, { ...headers, 'Content-Type': types[extname(path)] ?? 'application/octet-stream' });
    response.end(body);
  } catch {
    response.writeHead(404, headers);
    response.end('Not found');
  }
}).listen(port, () => console.log(`serve: http://localhost:${port}`));
