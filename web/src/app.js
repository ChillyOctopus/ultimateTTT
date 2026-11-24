// The page: the board, the menu cards, the bot, and the analysis. The engine (engine.wasm) is the one authority on the
// rules: this script keeps the list of moves and asks the engine what they add up to.
import { instantiate } from './engine.js';

/**
 * The bot's levels, weakest first. `settings` go to the engine's bot() (engine/lib.rs): Easy and Medium are
 * ultimattt's minimax, Hard and Expert uttt.ai's network and search (`visits`, capped at `ms`), with the solver
 * (`prove` ms) from move BOT_PROVE_FROM.
 */
const LEVELS = [
  { id: 'easy', name: 'Easy', note: 'Looks one move ahead, and slips up', settings: { depth: 1, blunder: 0.5 } },
  { id: 'medium', name: 'Medium', note: 'Looks three moves ahead', settings: { depth: 3, slack: 2 } },
  { id: 'hard', name: 'Hard', note: 'uttt.ai’s bot, 200 positions', settings: { visits: 200, ms: 2000, prove: 2000 } },
  { id: 'expert', name: 'Expert', note: 'uttt.ai’s bot, 1,000 positions', settings: { visits: 1000, ms: 6000, prove: 4000 } },
];
/** The strong levels vary their first move among the best few, so that games differ. */
const OPENING = { depth: 4, slack: 2 };
/** The analysis tries the solver from this many moves on: earlier positions take too long in a browser. */
const PROVE_FROM = 28;
const BOT_PROVE_FROM = 30;
/** The analysis's search: reported every THINK_STEP visits, up to THINK_MAX visits or THINK_BUDGET ms. */
const THINK_STEP = 100;
const THINK_MAX = 3000;
const THINK_BUDGET = 8000;
const PROVE_BUDGET = 10000;
const CHILD_BUDGET = 3000;
/** The shortest pause before the bot's move appears, so that a quick reply still reads as a reply. */
const BOT_PAUSE = 400;

const LINES = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]];
const $ = (id) => document.getElementById(id);

const stored = (() => {
  try {
    return JSON.parse(localStorage.getItem('uttt') ?? '{}');
  } catch {
    return {};
  }
})();
const settings = {
  level: LEVELS.some((l) => l.id === stored.level) ? stored.level : 'medium',
  side: stored.side === 'O' ? 'O' : 'X',
  showAnalysis: stored.showAnalysis === true,
};
const save = () => {
  try {
    localStorage.setItem('uttt', JSON.stringify(settings));
  } catch {
    // Private browsing: the settings just won't be remembered.
  }
};

const game = { mode: 'menu', moves: [], future: [], human: 'X', level: 'medium' };
let state = null; // engine.state(game.moves)
let engine;
let module;
let net = null; // net.bin, uttt.ai's network, fetched the first time a level or the analysis needs it
let netLoading = null;

/** net.bin, with its progress in the status line while it downloads. */
function loadNet() {
  netLoading ??= (async () => {
    const response = await fetch('net.bin');
    if (!response.ok) throw new Error(`net.bin: ${response.status}`);
    // Compressed in transit, the length header is the compressed size: fall back to the file's own.
    const size = response.headers.get('content-encoding') ? NET_SIZE : Number(response.headers.get('content-length')) || NET_SIZE;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      $('status').textContent = `Loading the network (10 MB): ${Math.min(99, Math.floor((100 * received) / size))}%`;
    }
    net = await new Blob(chunks).arrayBuffer();
    renderStatus();
  })().catch((error) => {
    netLoading = null;
    throw error;
  });
  return netLoading;
}
/** net.bin's size, from web/tools/export_net.py, for the progress when the server compresses it. */
const NET_SIZE = 9988620;

// --- Workers: one for the bot, one for the analysis's search, one for the solver --------------------------------------

let jobId = 0;
const workers = {};
function worker(name, onMessage) {
  if (workers[name] && net && !workers[name].hasNet) stop(name); // started before the network arrived
  if (!workers[name]) {
    const w = new Worker('worker.js', { type: 'module' });
    w.postMessage({ module, net });
    w.hasNet = !!net;
    workers[name] = w;
  }
  // Each job brings its own handler: a kept worker must not answer an old job's.
  workers[name].onmessage = ({ data }) => onMessage(data);
  return workers[name];
}
function stop(name) {
  workers[name]?.terminate();
  delete workers[name];
}

// --- The board --------------------------------------------------------------------------------------------------------

const board = $('board');
const cells = [];
const minis = [];
for (let b = 0; b < 9; b++) {
  const mini = document.createElement('div');
  mini.className = 'mini';
  for (let c = 0; c < 9; c++) {
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'cell';
    cell.dataset.square = String(9 * b + c);
    cell.disabled = true;
    mini.append(cell);
    cells.push(cell);
  }
  board.append(mini);
  minis.push(mini);
}
board.addEventListener('click', (event) => {
  const cell = event.target.closest('.cell');
  if (cell && !cell.disabled) play(Number(cell.dataset.square));
});

const turnOf = (count) => (count % 2 === 0 ? 'X' : 'O');
const humansTurn = () => game.mode !== 'bot' || state.next === game.human;
const markSpan = (p) => {
  const span = document.createElement('span');
  span.className = `mark-${p.toLowerCase()}`;
  span.textContent = p;
  return span;
};

function render() {
  const over = state.result !== null;
  const legal = new Set(state.legal);
  const open = $('card').hidden && game.mode !== 'menu' && humansTurn();
  const last = game.moves.at(-1);
  board.classList.toggle('x-turn', !over && state.next === 'X');
  board.classList.toggle('o-turn', !over && state.next === 'O');
  const line = over && state.result !== 'draw' ? LINES.find((l) => l.every((b) => state.boards[b] === state.result)) : null;
  minis.forEach((mini, b) => {
    const result = state.boards[b];
    mini.classList.toggle('closed', result !== '.');
    mini.classList.toggle('won-x', result === 'X');
    mini.classList.toggle('won-o', result === 'O');
    mini.classList.toggle('active', !over && result === '.' && (state.board === null || state.board === b));
    mini.classList.toggle('line', !!line?.includes(b));
  });
  cells.forEach((cell, sq) => {
    const mark = state.cells[sq];
    cell.classList.toggle('x', mark === 'X');
    cell.classList.toggle('o', mark === 'O');
    cell.classList.toggle('last', sq === last);
    cell.classList.toggle('legal', open && legal.has(sq));
    cell.disabled = !(open && legal.has(sq));
    const name = mark === 'X' ? 'X' : mark === 'O' ? 'O' : 'empty';
    cell.setAttribute('aria-label', `Board ${Math.floor(sq / 9) + 1}, square ${(sq % 9) + 1}: ${name}`);
  });

  const turn = $('turn');
  turn.replaceChildren();
  if (over) {
    turn.append(state.result === 'draw' ? 'A draw' : markSpan(state.result), state.result === 'draw' ? '' : ' wins');
  } else {
    turn.append(`Move ${game.moves.length + 1} · `, markSpan(state.next), ' to play');
  }
  $('turn').parentElement.classList.toggle('o-turn', !over && state.next === 'O');
  $('undo').disabled = !canUndo();
  $('redo').disabled = game.future.length === 0 || game.mode === 'menu';
  renderStatus();
}

function renderStatus() {
  const over = state.result !== null;
  let text = '';
  if (game.mode === 'menu') text = '';
  else if (over) text = resultText().lead;
  else if (game.mode === 'bot' && !humansTurn()) text = 'The bot is thinking…';
  else {
    const who = game.mode === 'bot' ? 'Your move' : `${state.next} to play`;
    text = state.board === null ? `${who}: any open board.` : `${who}, in the shaded board.`;
  }
  $('status').textContent = text;
  const level = LEVELS.find((l) => l.id === game.level);
  $('analysis-toggle-wrap').hidden = game.mode === 'analysis' || game.mode === 'menu';
  $('again-btn').hidden = game.mode === 'menu';
  $('menu-btn').hidden = game.mode === 'menu';
  $('again-btn').textContent = game.mode === 'bot' ? `New game (${level.name})` : 'New game';
}

// --- Moves ------------------------------------------------------------------------------------------------------------

function update() {
  state = engine.state(game.moves);
  stop('think');
  stop('prove');
  clearAnalysis();
  render();
  if (game.mode === 'menu') return;
  if (state.result !== null) {
    if (game.mode !== 'analysis') showResult();
  } else if (game.mode === 'bot' && !humansTurn()) {
    botMove();
  }
  if (analysisShown() && humansTurn()) analyze();
}

function play(square) {
  if (!state.legal.includes(square)) return;
  game.moves.push(square);
  game.future = [];
  update();
}

function canUndo() {
  if (game.mode === 'menu') return false;
  if (game.mode === 'bot') return game.moves.length > (game.human === 'X' ? 0 : 1);
  return game.moves.length > 0;
}

function undo() {
  if (!canUndo()) return;
  stop('bot');
  do game.future.push(game.moves.pop());
  while (game.mode === 'bot' && turnOf(game.moves.length) !== game.human);
  hideCard();
  update();
}

function redo() {
  if (game.future.length === 0) return;
  stop('bot');
  do game.moves.push(game.future.pop());
  while (game.mode === 'bot' && game.future.length && turnOf(game.moves.length) !== game.human);
  update();
}

async function botMove() {
  const id = ++jobId;
  const moves = [...game.moves];
  const level = LEVELS.find((l) => l.id === game.level);
  const strong = level.settings.visits && moves.length < 2;
  if (level.settings.visits && !net) {
    try {
      await loadNet();
    } catch (error) {
      $('status').textContent = 'The network didn’t load. Check your connection, then start a new game.';
      return;
    }
    if (id !== jobId || game.moves.join() !== moves.join()) return;
  }
  const settings = strong ? OPENING : { ...level.settings, prove: moves.length >= BOT_PROVE_FROM ? level.settings.prove : 0 };
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  const started = performance.now();
  worker('bot', (data) => {
    if (data.id !== jobId || game.moves.join() !== moves.join()) return;
    if (data.kind === 'error') return ($('status').textContent = `The bot hit an error: ${data.message}`);
    setTimeout(() => {
      if (data.id === jobId && game.moves.join() === moves.join()) play(data.move);
    }, Math.max(0, BOT_PAUSE - (performance.now() - started)));
  }).postMessage({ id, job: 'bot', moves, settings, seed });
}

// --- Analysis ---------------------------------------------------------------------------------------------------------

const analysisShown = () => game.mode === 'analysis' || (game.mode !== 'menu' && settings.showAnalysis);
let thought = null; // the latest round of the analysis's search: engine.think()
let proof = null; // { result, ms } for the position, once tried
let proofs = new Map(); // square → result, for each move once the position is proven

function clearAnalysis() {
  thought = null;
  proof = null;
  proofs = new Map();
  for (const cell of cells) {
    delete cell.dataset.q;
    delete cell.dataset.p;
    cell.removeAttribute('title');
  }
  $('analysis').hidden = !analysisShown() || game.mode === 'menu';
  $('verdict').textContent = '';
  $('proof').textContent = '';
  $('proof-legend').hidden = true;
  $('meter-x').style.transform = 'scaleX(0.5)';
}

async function analyze() {
  const moves = [...game.moves];
  const id = ++jobId;
  const same = (data) => data.id === id && game.moves.join() === moves.join();
  if (state.result === null) {
    $('verdict').textContent = 'Thinking…';
    try {
      await loadNet();
    } catch {
      $('verdict').textContent = 'The network didn’t load, so there’s no analysis. Check your connection.';
      return;
    }
    if (id !== jobId || game.moves.join() !== moves.join()) return;
    const seed = crypto.getRandomValues(new Uint32Array(1))[0];
    worker('think', (data) => {
      if (!same(data) || data.kind !== 'think') return;
      thought = data;
      renderAnalysis();
    }).postMessage({ id, job: 'think', moves, budget: THINK_BUDGET, maxVisits: THINK_MAX, step: THINK_STEP, seed });
  }
  if (moves.length >= PROVE_FROM && state.result === null) {
    $('proof').textContent = 'Solving…';
    worker('prove', (data) => {
      if (!same(data)) return;
      if (data.kind === 'prove') proof = data;
      if (data.kind === 'child') proofs.set(data.square, data.result);
      renderAnalysis();
    }).postMessage({ id, job: 'prove', moves, legal: state.legal, budget: PROVE_BUDGET, childBudget: CHILD_BUDGET });
  } else if (state.result === null) {
    $('proof').textContent = `From move ${PROVE_FROM}, the solver proves who wins with perfect play.`;
  }
  renderAnalysis();
}

/** A value, -1 to 1 for the side to move, as that side's expected score out of 100. */
const outOf100 = (v) => Math.round((v + 1) * 50);

function renderAnalysis() {
  if (!analysisShown()) return;
  const over = state.result !== null;
  const mover = state.next;
  // Each move against the search's pick (the one it would play): how much of its expected score it gives up.
  const rows = thought?.moves ?? [];
  const pick = rows.reduce((a, r) => (a && a[2] >= r[2] ? a : r), null);
  for (const [sq, visits, , mean] of rows) {
    if (mean === null) continue;
    const drop = pick[3] - mean;
    cells[sq].dataset.q = sq === pick[0] ? 'best' : drop <= 0.05 ? 'good' : drop <= 0.15 ? 'ok' : 'bad';
    cells[sq].title = `${mover} scores about ${outOf100(mean)} out of 100 after this move (${visits.toLocaleString('en-US')} positions searched)`;
  }
  for (const [sq, result] of proofs) {
    if (result === 'unknown') continue;
    cells[sq].dataset.p = result === 'draw' ? 'draw' : result === mover ? 'win' : 'loss';
    cells[sq].title = result === 'draw' ? 'Proven: draws' : result === mover ? 'Proven: wins' : 'Proven: loses';
  }
  $('proof-legend').hidden = proofs.size === 0;

  // The meter and the verdict, from X's side
  let share = 0.5;
  let verdict = '';
  const known = over ? state.result : proof && proof.result !== 'unknown' ? proof.result : null;
  if (known) {
    share = known === 'X' ? 1 : known === 'O' ? 0 : 0.5;
    verdict = over ? '' : known === 'draw' ? 'A draw with best play.' : `${known} wins with best play.`;
  } else if (thought) {
    const forX = mover === 'X' ? thought.value : -thought.value;
    share = 0.5 + 0.5 * forX;
    const leader = forX > 0 ? 'X' : 'O';
    const size = Math.abs(forX);
    const lean = size < 0.1 ? 'About even' : size < 0.35 ? `${leader} is a little better` : `${leader} is better`;
    verdict = `${lean}, after ${thought.visits.toLocaleString('en-US')} positions searched.`;
  }
  $('meter-x').style.transform = `scaleX(${share})`;
  $('verdict').textContent = verdict;

  if (proof) {
    const seconds = (proof.ms / 1000).toFixed(1);
    $('proof').textContent =
      proof.result === 'unknown'
        ? `The solver couldn’t settle this position in ${PROVE_BUDGET / 1000} seconds. It gets easier as the board fills up.`
        : `Proven in ${seconds} s, with perfect play from both sides.${proofs.size < state.legal.length ? ' Checking each move…' : ' Each move is marked with what it leads to.'}`;
  }
}

// --- Cards ------------------------------------------------------------------------------------------------------------

function showCard(templateId) {
  const card = $('card');
  card.replaceChildren($(templateId).content.cloneNode(true));
  card.hidden = false;
  if (state) render();
  return card;
}

function hideCard() {
  $('card').hidden = true;
}

function showMenu() {
  stop('bot');
  const resumable = game.mode !== 'menu' && game.moves.length > 0 && state.result === null;
  const card = showCard('menu-card');
  card.querySelector('[data-go="resume"]').hidden = !resumable;
  card.querySelector('.btn:not([hidden])').focus();
}

function showSetup() {
  const card = showCard('setup-card');
  const options = card.querySelector('[data-name="level"]');
  for (const level of LEVELS) {
    const label = document.createElement('label');
    const input = Object.assign(document.createElement('input'), { type: 'radio', name: 'level', value: level.id });
    input.checked = level.id === settings.level;
    const span = document.createElement('span');
    const name = document.createElement('b');
    name.textContent = level.name;
    span.append(name, level.note);
    label.append(input, span);
    options.append(label);
  }
  card.querySelector(`input[name="side"][value="${settings.side}"]`).checked = true;
  card.querySelector('input:checked').focus();
}

function resultText() {
  const r = state.result;
  if (game.mode === 'bot') {
    if (r === 'draw') return { title: 'A draw', lead: 'Neither side got three boards in a row.' };
    if (r === game.human) {
      return { title: 'You win!', lead: game.level === 'expert' ? 'You beat the strongest level.' : 'Well played. Try the next level up?' };
    }
    return { title: 'The bot wins', lead: 'Review the game to see where it turned.' };
  }
  if (r === 'draw') return { title: 'A draw', lead: 'Neither side got three boards in a row.' };
  return { title: `${r} wins`, lead: `${r} took three boards in a row.` };
}

function showResult() {
  const card = showCard('result-card');
  const { title, lead } = resultText();
  card.querySelector('[data-slot="title"]').textContent = title;
  card.querySelector('[data-slot="lead"]').textContent = lead;
  card.querySelector('.btn').focus();
}

function newGame(mode, moves = []) {
  stop('bot');
  game.mode = mode;
  game.moves = moves;
  game.future = [];
  game.human = settings.side;
  game.level = settings.level;
  hideCard();
  update();
}

$('card').addEventListener('click', (event) => {
  const go = event.target.closest('[data-go]')?.dataset.go;
  if (!go) return;
  if (go === 'setup') showSetup();
  else if (go === 'menu') showMenu();
  else if (go === 'resume') {
    hideCard();
    update();
  } else if (go === 'two' || go === 'analysis') newGame(go);
  else if (go === 'start') {
    const card = $('card');
    settings.level = card.querySelector('input[name="level"]:checked').value;
    settings.side = card.querySelector('input[name="side"]:checked').value;
    save();
    newGame('bot');
  } else if (go === 'again') newGame(game.mode);
  else if (go === 'review') newGame('analysis', [...game.moves]);
});

$('undo').addEventListener('click', undo);
$('redo').addEventListener('click', redo);
$('menu-btn').addEventListener('click', showMenu);
$('again-btn').addEventListener('click', () => newGame(game.mode));
$('analysis-toggle').checked = settings.showAnalysis;
$('analysis-toggle').addEventListener('change', (event) => {
  settings.showAnalysis = event.target.checked;
  save();
  update();
});
document.addEventListener('keydown', (event) => {
  if (event.target.closest('input, textarea') || event.altKey || event.ctrlKey || event.metaKey) return;
  if (event.key === 'ArrowLeft') undo();
  if (event.key === 'ArrowRight') redo();
});

// --- Start ------------------------------------------------------------------------------------------------------------

try {
  module = await WebAssembly.compileStreaming(fetch('engine.wasm')).catch(async () =>
    WebAssembly.compile(await (await fetch('engine.wasm')).arrayBuffer()),
  );
  engine = await instantiate(module);
  state = engine.state([]);
  render();
  showMenu();
} catch (error) {
  $('turn').textContent = 'The game couldn’t load in this browser.';
  console.error(error);
}
