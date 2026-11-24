// Assembles web/dist, the static site at uttt.jistevenson.com:
//
//   (cd web && npm ci && node build.mjs)
//
//   src/*          the page: HTML, CSS, the UI script, the worker and the engine wrapper
//   engine.wasm    engine/lib.rs (ultimattt's rules, minimax player and solver, and uttt.ai's
//                  network and search), built for WebAssembly
//   net.bin        uttt.ai's network's weights (src/net.bin, written by tools/export_net.py)
//   fonts/         Newsreader and Hanken Grotesk, as on jistevenson.com, from node_modules
//   licenses.txt   what the page carries from others, and the Apache License 2.0's text
//
// Needs Rust with the wasm32-unknown-unknown target (rustup target add wasm32-unknown-unknown).
import { execFileSync } from 'node:child_process';
import { copyFileSync, cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';

const web = new URL('./', import.meta.url).pathname;
const dist = `${web}dist/`;
const engine = `${web}engine/`;

execFileSync('cargo', ['build', '--release', '--target', 'wasm32-unknown-unknown'], { cwd: engine, stdio: 'inherit' });

rmSync(dist, { recursive: true, force: true });
cpSync(`${web}src/`, dist, { recursive: true });
copyFileSync(`${engine}target/wasm32-unknown-unknown/release/uttt_web.wasm`, `${dist}engine.wasm`);
mkdirSync(`${dist}fonts`);
const fonts = {
  '@fontsource-variable/newsreader/files/newsreader-latin-wght-normal.woff2': 'newsreader.woff2',
  '@fontsource-variable/hanken-grotesk/files/hanken-grotesk-latin-wght-normal.woff2': 'hanken-grotesk.woff2',
};
for (const [from, to] of Object.entries(fonts)) copyFileSync(`${web}node_modules/${from}`, `${dist}fonts/${to}`);
writeFileSync(
  `${dist}licenses.txt`,
  `This page carries work by others:

- engine.wasm includes ultimattt by Nelson Elhage (https://github.com/nelhage/ultimattt),
  modified, under the Apache License 2.0.
- net.bin is uttt.ai's policy-value network by Arkadiusz Nowaczyński
  (https://github.com/arnowaczynski/utttai), converted to float16, and engine.wasm
  includes a port of its search, under the Apache License 2.0.
- fonts/ holds Newsreader and Hanken Grotesk, under the SIL Open Font License 1.1.

The rest is by Jacob Stevenson, under the MIT License:
https://github.com/ChillyOctopus/ultimateTTT

The Apache License 2.0 follows.

${readFileSync(`${web}LICENSE-uttt.ai`, 'utf8')}`,
);
console.log('build: web/dist');
