// Runs the engine off the page's thread: the bot's moves, the analysis's search, and the solver. The page stops a
// job by ending the worker and starting another, since a search can't be interrupted.
import { instantiate } from './engine.js';

let engine;
let queue = Promise.resolve();
self.onmessage = ({ data }) => {
  queue = queue.then(() => handle(data));
};

const post = (id, kind, body) => self.postMessage({ id, kind, ...body });

async function handle(data) {
  if (data.module) {
    engine = await instantiate(data.module);
    if (data.net) engine.loadNet(new Uint8Array(data.net));
    return;
  }
  const { id, job } = data;
  try {
    if (job === 'bot') post(id, 'bot', engine.bot(data.moves, data.settings, data.seed));
    else if (job === 'think') think(id, data);
    else if (job === 'prove') prove(id, data);
  } catch (error) {
    post(id, 'error', { message: error.message });
  }
}

/** uttt.ai's search of the position, reported every `step` visits, until `maxVisits` or `budget` ms. */
function think(id, { moves, budget, maxVisits, step, seed }) {
  const start = performance.now();
  for (let visits = step; ; visits += step) {
    const round = engine.think(moves, visits, seed);
    post(id, 'think', round);
    if (round.visits >= maxVisits || performance.now() - start > budget) break;
  }
  post(id, 'done', {});
}

/** The position's result with best play; once that's proven, each legal move's. */
function prove(id, { moves, legal, budget, childBudget }) {
  const root = engine.prove(moves, budget);
  post(id, 'prove', root);
  if (root.result === 'unknown') return;
  for (const square of legal) post(id, 'child', { square, ...engine.prove([...moves, square], childBudget) });
  post(id, 'done', {});
}
