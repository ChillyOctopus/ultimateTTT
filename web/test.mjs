// Checks the built engine (dist/engine.wasm) through the same wrapper the page uses:
// the rules against the Rust engine's own records of 7 games, the bot's moves, and
// the solver against positions whose results the benchmark suite established.
//   node test.mjs
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { instantiate } from './src/engine.js';

const engine = await instantiate(readFileSync(new URL('dist/engine.wasm', import.meta.url)));
const data = new URL('../ultimattt/data/', import.meta.url);
let checks = 0;

// A game's positions, in ultimattt's notation, one per line: "X;..@......;<9 boards>".
const parse = (line) => {
  const [next, global, locals] = line.split(';');
  return { next, global, cells: locals.split('/').join('') };
};
const gameMoves = (positions) =>
  positions.slice(1).map((p, i) => {
    const before = positions[i].cells;
    const sq = [...p.cells].findIndex((c, j) => c !== before[j]);
    assert.ok(sq >= 0, 'every position adds a mark');
    return sq;
  });
const games = {};
for (const file of readdirSync(data).filter((f) => f.endsWith('-positions.txt'))) {
  const positions = readFileSync(new URL(file, data), 'utf8').trim().split('\n').map(parse);
  games[file.replace('-positions.txt', '')] = gameMoves(positions);

  // The rules: every position of the game, square by square.
  const moves = games[file.replace('-positions.txt', '')];
  positions.forEach((p, ply) => {
    const s = engine.state(moves.slice(0, ply));
    assert.equal(s.cells, p.cells, `${file} ply ${ply}: cells`);
    if (s.result === null) {
      assert.equal(s.next, p.next, `${file} ply ${ply}: player to move`);
      assert.equal(s.board, p.global.includes('@') ? p.global.indexOf('@') : null, `${file} ply ${ply}: board to play`);
      if (ply < moves.length) assert.ok(s.legal.includes(moves[ply]), `${file} ply ${ply}: the game's move is legal`);
    }
    checks++;
  });
}
assert.equal(Object.keys(games).length, 7);

// Illegal moves are refused, with a reason.
// After the center square (40), the reply must go in the center board (squares 36–44).
assert.throws(() => engine.state([40, 0]), /move 2 \(square 0\) is not legal/);
assert.throws(() => engine.state([81]), /not a square/);

// The bot plays legal moves, at every kind of setting.
const opening = games['selfplay-1s'].slice(0, 20);
for (const settings of [{ depth: 1, slack: 30 }, { depth: 1, blunder: 1 }, { depth: 3 }, { ms: 50 }]) {
  const { move } = engine.bot(opening, settings, 12345);
  assert.ok(engine.state(opening).legal.includes(move), `bot ${JSON.stringify(settings)} plays a legal move`);
  checks++;
}

// uttt.ai's network gives onnxruntime's answers (tools/net-fixture.json, from tools/export_net.py), to float16's precision.
engine.loadNet(readFileSync(new URL('src/net.bin', import.meta.url)));
for (const { moves, value, policy } of JSON.parse(readFileSync(new URL('tools/net-fixture.json', import.meta.url)))) {
  const got = engine.net(moves);
  assert.ok(Math.abs(got.value - value) < 0.005, `net value after ${moves.length} moves: ${got.value} vs ${value}`);
  assert.equal(got.policy.length, Object.keys(policy).length);
  for (const [sq, p] of got.policy) assert.ok(Math.abs(p - policy[sq]) < 0.005, `net policy at square ${sq}`);
  checks++;
}

// The search plays legal moves, and the analysis's search gives every legal move a value.
for (const settings of [{ visits: 50 }, { visits: 1000, ms: 100 }]) {
  const { move } = engine.bot(opening, settings, 7);
  assert.ok(engine.state(opening).legal.includes(move), `bot ${JSON.stringify(settings)} plays a legal move`);
  checks++;
}
const thought = engine.think(opening, 200, 7);
assert.ok(thought.visits >= 200);
assert.deepEqual(thought.moves.map(([sq]) => sq).sort((a, b) => a - b), engine.state(opening).legal.slice().sort((a, b) => a - b));
assert.ok(thought.moves.every(([, visits, , mean]) => visits > 0 && Math.abs(mean) <= 1));
checks++;

// The solver, against results from bench/suite.tsv in the research repo (True: the side to move can force a win).
for (const [game, ply, wins] of [['random1', 43, true], ['random1', 46, false], ['random2', 50, true], ['random2', 53, false], ['selfplay-1s', 53, false]]) {
  const moves = games[game].slice(0, ply);
  const mover = engine.state(moves).next;
  const { result, ms } = engine.prove(moves, 60000);
  assert.notEqual(result, 'unknown', `${game}-${ply} is solved`);
  assert.equal(result === mover, wins, `${game}-${ply}: ${result} (${ms} ms)`);
  console.log(`  ${game}-${ply}: ${result} in ${ms} ms`);
  checks++;

  // With the solver, the bot keeps a proven win or draw: the move it plays proves to the same result.
  const reply = engine.bot(moves, { visits: 20, prove: 60000 }, 7);
  if (result === mover || result === 'draw') {
    assert.equal(reply.proven, result, `${game}-${ply}: the bot keeps ${result}`);
    assert.equal(engine.prove([...moves, reply.move], 60000).result, result, `${game}-${ply}: its move keeps ${result}`);
  } else {
    assert.equal(reply.proven, undefined, `${game}-${ply}: lost, so nothing to keep`);
  }
  checks++;
}

console.log(`test: ${checks} checks passed`);
