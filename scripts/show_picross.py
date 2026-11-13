"""Visualize a Picross puzzle, its line agents / graph, and the line-logic reference solver.

No model needed: this shows what the Sheaf-ADMM model will receive and what it has to do.

* ``puzzle.png``: the clues alone (the input) and the unique solution (the label).
* ``agents.png``: one row agent and one column agent on the board, the ``K(n, n)`` agent graph
  with their edges, and their ``n``-cell views with the shared cell (edge slots
  ``map_u`` / ``map_v``), checked against ``views.build_picross_graph``.
* ``propagation.png``: parallel line-logic rounds from the empty board (every line deduces from
  the same state, like one synchronous ADMM / message-passing round). On a ``test_hard`` board
  it stalls with grey cells left: line logic alone is not enough.

    uv run python scripts/show_picross.py --split test_easy --index 0 --row 3 --col 6
    uv run python scripts/show_picross.py --split test_hard --index 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sheaf_admm.data import picross as P
from sheaf_admm.data import views as V
from sheaf_admm.data.loaders import PuzzleDataset
from sheaf_admm.viz._render import PICROSS_COLORS, render_picross_panel, save_figure

ROW_COLOR, COL_COLOR, SHARED_COLOR = "#2563eb", "#f59e0b", "#a21caf"


def load_board(data_dir: Path, split: str, index: int):
    s = PuzzleDataset(data_dir, split).sets["all"]
    n = int(round(np.sqrt(s.labels.shape[1])))
    x, y = np.array(s.inputs[index]), np.array(s.labels[index]).reshape(n, n)
    rounds_file = data_dir / split / "all__rounds.npy"
    rounds = int(np.load(rounds_file, mmap_mode="r")[index]) if rounds_file.exists() else None
    return n, x, y, P.unpack_clues(x.reshape(2 * n, P.clue_slots(n))), rounds


def plot_puzzle(clues, y, title, out):
    n = y.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    render_picross_panel(axes[0], clues, np.full((n, n), -1))
    axes[0].set_title("input: clues (grey = unknown)")
    render_picross_panel(axes[1], clues, y)
    axes[1].set_title("label: unique solution")
    fig.suptitle(title)
    save_figure(fig, out)


def _fmt(clue) -> str:
    return " ".join(map(str, clue)) or "0"


def _strip(ax, values, highlight, color, label):
    n = len(values)
    for k, v in enumerate(values):
        ax.add_patch(
            plt.Rectangle((k, 0), 1, 1, color=PICROSS_COLORS["filled" if v else "empty"], zorder=0)
        )
        ax.text(k + 0.5, -0.15, str(k), ha="center", va="bottom", fontsize=7, color="#6b7280")
    ax.add_patch(
        plt.Rectangle((highlight, 0), 1, 1, color=SHARED_COLOR, alpha=0.5, lw=0, zorder=1)
    )
    for k in range(n + 1):
        ax.plot([k, k], [0, 1], color="black", lw=0.5, zorder=2)
    ax.plot([0, n], [0, 0], color=color, lw=3, zorder=2)
    ax.plot([0, n], [1, 1], color=color, lw=3, zorder=2)
    ax.text(-0.3, 0.5, label, ha="right", va="center", fontsize=9, color=color)
    ax.set_xlim(-0.3, n)
    ax.set_ylim(1.4, -0.7)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_agents(clues, y, row, col, out):
    n = y.shape[0]
    edges, map_u, map_v = V.build_picross_graph(n)
    e = row * n + col
    u, v = (int(t) for t in edges[e])
    assert (u, v) == (row, n + col) and int(map_u[e]) == col and int(map_v[e]) == row

    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 0.9, 1.3], height_ratios=[1, 1])

    ax = fig.add_subplot(gs[:, 0])
    row_mask = np.zeros((n, n), bool)
    row_mask[row] = True
    col_mask = np.zeros((n, n), bool)
    col_mask[:, col] = True
    shared = row_mask & col_mask
    render_picross_panel(
        ax, clues, y,
        overlays=[(row_mask & ~shared, ROW_COLOR), (col_mask & ~shared, COL_COLOR),
                  (shared, SHARED_COLOR)],
    )
    ax.set_title(f"row agent {row} (blue), column agent {n + col} (orange)\n"
                 f"shared cell ({row}, {col}) = edge {e}")

    ax = fig.add_subplot(gs[:, 1])
    ys = np.arange(n)
    for (a, b) in edges.tolist():
        ax.plot([0, 1], [a, b - n], color="#9ca3af", lw=0.4, alpha=0.5, zorder=0)
    for j in range(n):
        ax.plot([0, 1], [row, j], color=ROW_COLOR, lw=1.0, alpha=0.8, zorder=1)
    for i in range(n):
        ax.plot([0, 1], [i, col], color=COL_COLOR, lw=1.0, alpha=0.8, zorder=1)
    ax.plot([0, 1], [row, col], color=SHARED_COLOR, lw=3, zorder=2)
    ax.scatter(np.zeros(n), ys, s=110, color=ROW_COLOR, zorder=3)
    ax.scatter(np.ones(n), ys, s=110, color=COL_COLOR, zorder=3)
    for k in range(n):
        ax.text(-0.08, k, f"row {k}  (agent {k})", ha="right", va="center", fontsize=7)
        ax.text(1.08, k, f"(agent {n + k})  col {k}", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.75, 1.75)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axis("off")
    ax.set_title(f"agent graph K({n},{n}): {2 * n} agents, {n * n} edges (1 per cell)")

    slices = V.picross_slice_batch(y[None, :, :, None])[0, :, :, 0].numpy()  # [2n, n]
    ax = fig.add_subplot(gs[0, 2])
    _strip(ax, slices[row], col, ROW_COLOR, f"agent {row}\nclue {_fmt(clues[row])}")
    ax.set_title(f"agent views (slot index on top); edge {e}: map_u = {col}", fontsize=9)
    ax = fig.add_subplot(gs[1, 2])
    _strip(ax, slices[n + col], row, COL_COLOR, f"agent {n + col}\nclue {_fmt(clues[n + col])}")
    ax.set_title(f"map_v = {row}", fontsize=9)
    save_figure(fig, out)


def propagation_history(clues, n):
    cands = [P.line_placements(c, n) for c in clues]
    state = np.full((n, n), P.UNKNOWN, dtype=np.int8)
    states = [state.copy()]
    while True:
        res = P.propagate(state, cands, parallel=True, max_rounds=1)
        if res is None:
            raise RuntimeError("contradiction: the clues have no solution")
        state, cands, rounds = res
        if rounds == 0:
            return states
        states.append(state.copy())


def plot_propagation(clues, y, out, cols=6):
    n = y.shape[0]
    states = propagation_history(clues, n)
    total = len(states)
    rows = int(np.ceil(total / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for k, ax in enumerate(axes.flat):
        if k >= total:
            ax.axis("off")
            continue
        s = states[k]
        fresh = (s != P.UNKNOWN) & (states[k - 1] == P.UNKNOWN) if k else np.zeros_like(s, bool)
        render_picross_panel(ax, clues, s, overlays=[(fresh, "#22c55e")], clue_fontsize=4)
        known = int((s != P.UNKNOWN).sum())
        ax.set_title(f"round {k}: {known}/{n * n} known (+{int(fresh.sum())})", fontsize=8)
    final = states[-1]
    done = (final != P.UNKNOWN).all()
    assert (final[final != P.UNKNOWN] == y[final != P.UNKNOWN]).all(), "deduction contradicts label"
    left = int((final < 0).sum())
    verdict = "solved by line logic" if done else f"stalls with {left} unknown cells"
    fig.suptitle(f"parallel line-logic rounds (green = fixed this round): {verdict}")
    save_figure(fig, out)
    return len(states) - 1, done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=Path("datasets/picross_10x10"))
    p.add_argument("--split", default="test_easy")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--row", type=int, default=3, help="row agent to highlight")
    p.add_argument("--col", type=int, default=6, help="column to highlight")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/viz_picross"))
    a = p.parse_args()

    n, _, y, clues, rounds = load_board(a.data_dir, a.split, a.index)
    out = a.out_dir / f"{a.split}_{a.index}"
    out.mkdir(parents=True, exist_ok=True)
    title = f"{a.split}[{a.index}]  ({n}x{n}, stored rounds = {rounds})"
    plot_puzzle(clues, y, title, out / "puzzle.png")
    plot_agents(clues, y, a.row, a.col, out / "agents.png")
    used, done = plot_propagation(clues, y, out / "propagation.png")
    print(f"{title}: {used} rounds, {'solved' if done else 'stalled'} -> {out}/")


if __name__ == "__main__":
    main()
