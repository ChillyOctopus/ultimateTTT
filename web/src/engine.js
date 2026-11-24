// The engine (engine/lib.rs, built to engine.wasm), wrapped: each call writes
// a request into the engine's buffer and parses the JSON answer. Runs in the
// page, in the worker, and in Node for the tests.

export async function instantiate(module) {
  // Bytes give { module, instance }; a compiled module gives the instance alone.
  const made = await WebAssembly.instantiate(module, {
    env: { now_ms: () => performance.now() },
  });
  const { exports } = made.instance ?? made;
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();
  const call = (name, ...parts) => {
    const bytes = encoder.encode(parts.join(';'));
    new Uint8Array(exports.memory.buffer, exports.buffer(), bytes.length).set(bytes);
    const length = exports[name](bytes.length);
    // Read the buffer only after the call: memory may have grown, which replaces the old view.
    const answer = JSON.parse(decoder.decode(new Uint8Array(exports.memory.buffer, exports.buffer(), length)));
    if (answer.error) throw new Error(answer.error);
    return answer;
  };
  const moves = (list) => list.join(' ');
  /** Copy in the network's weights, net.bin. */
  const loadNet = (bytes) => {
    const at = exports.net_buffer(bytes.length);
    new Uint8Array(exports.memory.buffer, at, bytes.length).set(bytes);
    const length = exports.net_load();
    const answer = JSON.parse(decoder.decode(new Uint8Array(exports.memory.buffer, exports.buffer(), length)));
    if (answer.error) throw new Error(answer.error);
  };
  return {
    loadNet,
    /** The board after `list` (squares 0–80): cells, boards, next, board, legal, result. */
    state: (list) => call('state', moves(list)),
    /** The bot's move: { move, … } (see bot() in engine/lib.rs for the settings). */
    bot: (list, { depth = 0, ms = 0, slack = 0, blunder = 0, visits = 0, prove = 0 }, seed) =>
      call('bot', moves(list), depth, ms, slack, blunder, seed, visits, prove),
    /** The network alone: { value, policy: [[square, probability], …] }. */
    net: (list) => call('net', moves(list)),
    /** uttt.ai's search, carried on to `visits`: { visits, value, moves: [[square, visits, chance, mean, network], …] }. */
    think: (list, visits, seed) => call('think', moves(list), visits, seed),
    /** The result with best play, proven within about `ms`: { result: 'X' | 'O' | 'draw' | 'unknown', ms }. */
    prove: (list, ms) => call('prove', moves(list), ms),
  };
}
