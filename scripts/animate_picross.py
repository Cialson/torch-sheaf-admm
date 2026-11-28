"""Animate how a trained Sheaf-ADMM model solves one Picross board, next to line logic.

Each frame t shows, side by side:

* left: the state after t parallel line-logic rounds (``data.picross.propagate``), grey = not
  deduced yet; on a ``test_hard`` board it stalls with grey cells left;
* right: the Sheaf prediction after t ADMM iterations (the model run with ``num_iters=t``,
  last x-iterate decoded, row/column views averaged). Cells with a filled probability between
  ``--unsure`` and ``1 - --unsure`` are drawn grey (undecided), wrong decided cells are outlined
  in red.

The two step counts are different units (a round of human-style deduction vs one ADMM
iteration); on ``test_easy`` the model needs about 1-2 iterations per round (see
``analyze_picross.py``).

    uv run python scripts/animate_picross.py outputs/<run>/checkpoint.pt --split test_easy --index 2
    uv run python scripts/animate_picross.py outputs/<run>/checkpoint.pt --split test_hard --index 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_picross import final_logits, load_model  # noqa: E402

from sheaf_admm.data import PuzzleDataset  # noqa: E402
from sheaf_admm.data import picross as P  # noqa: E402
from sheaf_admm.data import views as V  # noqa: E402
from sheaf_admm.training.loop import move_to_device  # noqa: E402
from sheaf_admm.viz._render import render_picross_panel  # noqa: E402


def load_board(data_dir: Path, split: str, index: int):
    s = PuzzleDataset(data_dir, split).sets["all"]
    x = np.array(s.inputs[index : index + 1])
    y = np.array(s.labels[index : index + 1])
    n = int(round(np.sqrt(y.shape[1])))
    batch = {"inputs": x, "labels": y, "height": n, "width": n}
    clues = P.unpack_clues(x[0].reshape(2 * n, P.clue_slots(n)))
    return batch, clues, y[0].reshape(n, n), n


def line_logic_states(clues, n, steps):
    """State after 0..steps parallel rounds (the last state repeats once it is stuck)."""
    cands = [P.line_placements(c, n) for c in clues]
    state = np.full((n, n), P.UNKNOWN, dtype=np.int8)
    states = [state.copy()]
    for _ in range(steps):
        res = P.propagate(state, cands, parallel=True, max_rounds=1)
        if res is None:
            raise RuntimeError("contradiction: the clues have no solution")
        state, cands, _ = res
        states.append(state.copy())
    return states


@torch.no_grad()
def sheaf_probs(model, task, batch, steps, device):
    """Filled-cell probability ``[steps + 1, n, n]`` after 0..steps ADMM iterations
    (step 0 = nothing computed yet, 0.5 everywhere)."""
    fwd, _, _ = task.prepare(batch)
    fwd = move_to_device(fwd, device)
    n = batch["height"]
    out = [np.full((n, n), 0.5)]
    for k in range(1, steps + 1):
        views = final_logits(model, "sheaf", fwd, k).permute(1, 0, 2, 3)  # [1, 2n, n, 2]
        probs = V.reassemble_picross_logits(views.softmax(-1))[0, ..., 1]
        out.append(probs.cpu().numpy())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--split", default="test_easy")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--steps", type=int, default=20, help="last iteration / round shown")
    p.add_argument("--unsure", type=float, default=0.2, help="grey if p in [u, 1 - u]")
    p.add_argument("--fps", type=float, default=2.0)
    p.add_argument("--hold", type=int, default=6, help="extra frames on the last step")
    p.add_argument("--params", choices=("ema", "live"), default="ema")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis_picross/gif"))
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, task, cfg, trained, _ = load_model(a.checkpoint, a.params, device)
    if cfg.model_type != "sheaf":
        raise SystemExit("only Sheaf checkpoints have ADMM iterations to animate")
    batch, clues, label, n = load_board(Path(cfg.data.dir), a.split, a.index)
    logic = line_logic_states(clues, n, a.steps)
    probs = sheaf_probs(model, task, batch, a.steps, device)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 5.0))
    kind = "facile" if a.split == "test_easy" else "difficile"
    fig.suptitle(f"Picross {kind} ({a.split}[{a.index}]) : déduction humaine vs Sheaf-ADMM",
                 fontsize=13)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.02, wspace=0.08)

    def draw(t):
        t = min(t, a.steps)
        for ax in axes:
            ax.clear()
        s = logic[t]
        known = int((s != P.UNKNOWN).sum())
        render_picross_panel(axes[0], clues, s)
        stuck = t > 0 and np.array_equal(logic[t], logic[t - 1]) and known < n * n
        axes[0].set_title(f"propagation ligne par ligne, round {t}\n{known}/{n * n} cases "
                          f"déduites" + ("  (bloquée)" if stuck else ""), fontsize=10)

        pr = probs[t]
        state = np.where(pr > 1 - a.unsure, 1, np.where(pr < a.unsure, 0, -1))
        pred = (pr > 0.5).astype(int)
        wrong = (state >= 0) & (pred != label)
        render_picross_panel(axes[1], clues, state, wrong=wrong)
        solved = t > 0 and (pred == label).all()
        unsure = int((state < 0).sum())
        if t == 0:
            status = "rien de calculé"
        elif solved:
            status = "solution trouvée" + (f" ({unsure} cases encore peu sûres)" if unsure else "")
        else:
            status = f"{int(wrong.sum())} fausses (en rouge), {unsure} incertaines (grises)"
        axes[1].set_title(f"Sheaf-ADMM, itération {t} (entraîné à K={trained})\n{status}",
                          fontsize=10, color="#15803d" if solved else "black")

    frames = list(range(a.steps + 1)) + [a.steps] * a.hold
    anim = FuncAnimation(fig, draw, frames=frames)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.split}_{a.index}.gif"
    anim.save(out, writer=PillowWriter(fps=a.fps), dpi=90)
    plt.close(fig)

    first = next((k for k in range(1, a.steps + 1)
                  if ((probs[k] > 0.5).astype(int) == label).all()), None)
    logic_done = next((r for r, s in enumerate(logic) if (s != P.UNKNOWN).all()), None)
    print(f"{out}: Sheaf solved at iteration {first}, line logic "
          f"{'complete at round ' + str(logic_done) if logic_done is not None else 'stalls'}")


if __name__ == "__main__":
    main()
