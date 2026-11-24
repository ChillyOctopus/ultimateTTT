# Ultimate tic-tac-toe

Nine games of tic-tac-toe in one: play a bot, play a friend on the same device, or study a position on an analysis
board that proves late positions won, drawn, or lost.

**Play it at [uttt.jistevenson.com](https://uttt.jistevenson.com).** Everything runs in your browser as WebAssembly,
so your moves never leave your device.

## How it works

| Part | Where | What it does |
|---|---|---|
| Rules and solver | `ultimattt/` | Nelson Elhage's [ultimattt](https://github.com/nelhage/ultimattt) in Rust: the rules, a minimax player, and a proof-number solver, ported to stable Rust and WebAssembly, with two endgame rules corrected |
| Network | `web/engine/net.rs`, `web/src/net.bin` | [uttt.ai](https://github.com/arnowaczynski/utttai)'s policy-value network, Arkadiusz Nowaczyński's AlphaZero-style player, evaluated in plain Rust from float16 weights |
| Search | `web/engine/mcts.rs` | uttt.ai's Monte Carlo tree search, guided by the network, as on its website |
| Engine | `web/engine/lib.rs` | The WebAssembly module the page calls: the board, the bot's move, the analysis's search, and the solver |
| Page | `web/src/` | The static site: the board, the menus, and Web Workers that run the engine off the page's thread |
| Python | `python/` | The first version of the game, in Python: a terminal game with simple bots |

The levels: Easy and Medium are ultimattt's minimax, looking one and three moves ahead. Hard and Expert are uttt.ai's
search at 200 and 1,000 positions a move (1,000 is uttt.ai's own setting on a computer). From move 30, before each
move, Hard and Expert also run the solver: once it proves the position won or drawn, the bot plays a move that keeps
that result.

`web/tools/export_net.py` writes `net.bin` from uttt.ai's ONNX file and checks the Rust arithmetic, in numpy, against
onnxruntime; `web/test.mjs` checks the WebAssembly build against the same answers.

## Run it

Needs Rust (stable, with `rustup target add wasm32-unknown-unknown`) and Node 24.

```bash
cd web
npm ci
node build.mjs        # builds the engine and assembles web/dist
node test.mjs         # the rules on seven recorded games, the network, the bot, and the solver
node serve.mjs        # http://localhost:4331
```

The engine on its own, natively:

```bash
cd ultimattt
cargo test --release
```

`ultimattt/README.md` is the upstream project's README.

## License

MIT, except `ultimattt/` (Apache License 2.0, `ultimattt/LICENSE`) and uttt.ai's network and the ports of its code
(Apache License 2.0, `web/LICENSE-uttt.ai`). [LICENSE](LICENSE) has the details.
