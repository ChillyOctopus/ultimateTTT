"""Write uttt.ai's policy-value network (Apache-2.0, Arkadiusz Nowaczyński) as
web/src/net.bin for the engine in web/engine/net.rs.

    python3 export_net.py policy_value_net_stage2.onnx ../src/net.bin net-fixture.json

The ONNX file is the one uttt.ai serves (https://uttt.ai/policy_value_net_stage2.onnx).
Its batch norms are already folded into the convolutions. The weights are
stored as float16 (half the download), in the order and layout net.rs reads:
every matrix row-major as [output][input], activations as [square][channel].
The script then checks a numpy copy of net.rs's arithmetic against
onnxruntime on random positions, and saves 50 of them with onnxruntime's
answers for web/test.mjs. It needs onnx, onnxruntime and numpy.
"""
import json
import random
import sys

import numpy as np
import onnx
from onnx import numpy_helper

LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def load(path):
    model = onnx.load(path)
    w = {i.name: numpy_helper.to_array(i).astype(np.float32) for i in model.graph.initializer}
    convs = [n for n in model.graph.node if n.op_type in ("Conv", "ConvTranspose", "Gemm")]
    return [(n.op_type, w[n.input[1]], w[n.input[2]]) for n in convs]


def tensors(layers):
    """The network's weights in net.rs's order and layout."""
    out = []
    it = iter(layers)

    def take(kind):
        op, wt, b = next(it)
        assert op == kind, (op, kind)
        return wt, b

    def dense(wt, b):  # a 1x1 convolution or a Gemm: [out][in]
        out.extend([wt.reshape(wt.shape[0], -1), b])

    wt, b = take("Conv")  # 3x3 stride 3, 4 -> 128: [out][in*9 + ky*3 + kx]
    out.extend([wt.reshape(128, 36), b])
    dense(*take("Conv"))  # 128 -> 128
    dense(*take("Conv"))  # 128 -> 256
    for _ in range(3):
        dense(*take("Conv"))  # 256 -> 512
        wt, b = take("Conv")  # depthwise 3x3: [channel][square]
        out.extend([wt.reshape(512, 9), b])
        wt, b = take("ConvTranspose")  # [in][out][ky][kx] -> [square][out][in]
        out.extend([wt.transpose(2, 3, 1, 0).reshape(9 * 256, 512), b])
    dense(*take("Conv"))  # policy, 640 -> 256
    wt, b = take("ConvTranspose")  # upsampling: [cell][out][in]
    out.extend([wt.transpose(2, 3, 1, 0).reshape(9 * 256, 256), b])
    dense(*take("Conv"))  # 260 -> 128
    dense(*take("Conv"))  # 128 -> 1
    for _ in range(2):  # value, from x2 then x3
        dense(*take("Conv"))  # 256 -> 256
        wt, b = take("Conv")  # depthwise
        out.extend([wt.reshape(256, 9), b])
    dense(*take("Gemm"))  # 512 -> 128
    dense(*take("Gemm"))  # 128 -> 1
    assert next(it, None) is None
    return out


def elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def forward(t, planes):
    """net.rs's arithmetic. planes: [81 squares][4]. Returns (logits by square, value)."""
    t = iter(t)
    nx = lambda: next(t)
    w, b = nx(), nx()
    patches = planes.reshape(9, 9, 4).transpose(0, 2, 1).reshape(9, 36)  # [board][in*9+cell]
    x1 = elu(patches @ w.T + b)
    w, b = nx(), nx(); h = elu(x1 @ w.T + b)
    w, b = nx(), nx(); x2 = np.maximum(h @ w.T + b, 0)
    x = x2
    for _ in range(3):
        w, b = nx(), nx(); h = elu(x @ w.T + b)  # [9][512]
        w, b = nx(), nx(); d = elu((h * w.T).sum(0) + b)  # [512]
        w, b = nx(), nx(); up = (w @ d).reshape(9, 256) + b
        x = np.maximum(up + x, 0)
    x3 = x
    w, b = nx(), nx(); p = elu(np.concatenate([x1, x2, x3], 1) @ w.T + b)  # [9][256]
    w, b = nx(), nx()
    up = np.einsum("koi,bi->bko", w.reshape(9, 256, 256), p) + b  # [board][cell][256]
    up = elu(up.reshape(81, 256))
    w, b = nx(), nx(); h = elu(np.concatenate([planes, up], 1) @ w.T + b)
    w, b = nx(), nx(); logits = np.clip((h @ w.T + b)[:, 0], -32, 32)
    vs = []
    for src in (x2, x3):
        w, b = nx(), nx(); h = src @ w.T + b
        w, b = nx(), nx(); vs.append(elu((h * w.T).sum(0) + b))
    w, b = nx(), nx(); h = elu(np.concatenate(vs) @ w.T + b)
    w, b = nx(), nx(); v = np.tanh(h @ w.T + b)[0]
    return logits, float(v)


def random_position(rng):
    """(cells by square 0 empty/1 X/2 O, legal squares, X to move?, moves) after random moves."""
    cells, boards, board, player, moves = [0] * 81, [0] * 9, None, 1, []
    for _ in range(rng.randint(0, 60)):
        legal = legal_squares(cells, boards, board)
        if not legal:
            break
        sq = rng.choice(legal)
        moves.append(sq)
        cells[sq] = player
        b, c = divmod(sq, 9)
        mine = [cells[9 * b + i] == player for i in range(9)]
        if any(all(mine[i] for i in line) for line in LINES):
            boards[b] = player
            if any(all(boards[i] == player for i in line) for line in LINES):
                return cells, [], player == 1, moves  # won: the game is over
        elif all(cells[9 * b:9 * b + 9]):
            boards[b] = 3
        board = c if not boards[c] else None
        player = 3 - player
    return cells, legal_squares(cells, boards, board), player == 1, moves


def legal_squares(cells, boards, board):
    open_ = [b for b in range(9) if not boards[b]] if board is None else [board]
    return [9 * b + i for b in open_ for i in range(9) if not cells[9 * b + i]]


def planes_of(cells, legal, x_to_move):
    me = 1 if x_to_move else 2
    p = np.zeros((81, 4), np.float32)
    for sq, v in enumerate(cells):
        if v:
            p[sq, 0 if v == me else 1] = 1
    p[:, 2] = 1 if x_to_move else -1
    p[legal, 3] = 1
    return p


def onnx_input(planes):
    """[81 squares][4] -> ONNX's [1][4][9 rows][9 columns]."""
    grid = np.zeros((1, 4, 9, 9), np.float32)
    for sq in range(81):
        b, c = divmod(sq, 9)
        grid[0, :, 3 * (b // 3) + c // 3, 3 * (b % 3) + c % 3] = planes[sq]
    return grid


def main(src, dst, fixture):
    import onnxruntime as ort

    full = tensors(load(src))
    half = [a.astype(np.float16) for a in full]
    session = ort.InferenceSession(src)
    rng = random.Random(1)
    worst_logit, worst_value, agree, n, saved = 0.0, 0.0, 0, 0, []
    for _ in range(300):
        cells, legal, xm, moves = random_position(rng)
        if not legal:
            continue
        planes = planes_of(cells, legal, xm)
        pl, sv = session.run(None, {"input": onnx_input(planes)})
        ref = np.array([pl[0][3 * (s // 9 // 3) + s % 9 // 3, 3 * (s // 9 % 3) + s % 9 % 3] for s in range(81)])
        mine, v = forward([a.astype(np.float32) for a in half], planes)
        exact, ve = forward(full, planes)
        assert np.abs(exact - ref).max() < 1e-3 and abs(ve - sv[0]) < 1e-4, "layout mismatch"
        worst_logit = max(worst_logit, np.abs(mine[legal] - ref[legal]).max())
        worst_value = max(worst_value, abs(v - sv[0]))
        agree += legal[int(np.argmax(mine[legal]))] == legal[int(np.argmax(ref[legal]))]
        n += 1
        if len(saved) < 50:
            p = np.exp(ref[legal] - ref[legal].max())
            p /= p.sum()
            saved.append({"moves": moves, "value": round(float(sv[0]), 5),
                          "policy": {str(s): round(float(q), 5) for s, q in zip(legal, p)}})
    print(f"{n} positions: float32 layout matches onnxruntime; float16 worst logit error "
          f"{worst_logit:.4f}, worst value error {worst_value:.4f}, same top move {agree}/{n}")
    with open(dst, "wb") as f:
        f.write(b"UTTTNET1")
        for a in half:
            f.write(a.astype("<f2").tobytes())
    print("wrote", dst, sum(a.size for a in half), "weights")
    with open(fixture, "w") as f:
        json.dump(saved, f, separators=(",", ":"))
    print("wrote", fixture, len(saved), "positions")


if __name__ == "__main__":
    main(*sys.argv[1:4])
