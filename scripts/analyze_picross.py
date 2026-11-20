"""Compare Picross checkpoints board by board, across ADMM iterations / MPNN rounds.

For every checkpoint, split and iteration count K, records which boards are solved (and the
fraction of satisfied line clues), then relates it to each board's difficulty, i.e. the number
of parallel line-logic rounds stored by ``build_picross`` (``all__rounds.npy``):

* ``solved_vs_k.png``: solved rate vs K per model and split (trained K marked);
* ``solved_vs_rounds.png``: solved rate at a common K (default: the trained K, where the MPNN
  is at its best), by line-logic rounds;
* ``first_k_vs_rounds.png``: for boards a model solves, the smallest K that solves them, vs the
  line-logic rounds (do ADMM iterations / MPNN rounds play the role of propagation rounds?);
* ``results.json`` (tables) and ``per_board.npz`` (raw per-board arrays).

    uv run python scripts/analyze_picross.py outputs/A/checkpoint.pt outputs/B/checkpoint.pt \\
        --labels sheaf mpnn --out-dir outputs/analysis_picross
    # redraw the figures from the saved per-board results, without the GPU pass:
    uv run python scripts/analyze_picross.py outputs/A/checkpoint.pt outputs/B/checkpoint.pt \\
        --labels sheaf mpnn --out-dir outputs/analysis_picross --reuse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from sheaf_admm.data import PuzzleDataset
from sheaf_admm.data import views as V
from sheaf_admm.models import model_config_from_dict
from sheaf_admm.training import build_model, make_task
from sheaf_admm.training.loop import move_to_device
from sheaf_admm.training.tasks import picross_line_clues
from sheaf_admm.viz._render import save_figure

ROUND_BINS = [(0, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 10), (11, 99)]


def load_model(path: Path, params: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    if cfg.task != "picross":
        raise SystemExit(f"{path}: not a picross checkpoint (task={cfg.task})")
    task = make_task(cfg.task, **OmegaConf.to_container(cfg.task_cfg, resolve=True))
    model_cfg = model_config_from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    model = build_model(model_cfg, cfg.model_type, seed=0).to(device)
    state = ckpt.get("ema_state") if params == "ema" else None
    model.load_state_dict(state if state is not None else ckpt["model_state"])
    model.eval()
    t = cfg.training
    trained = t.K_train if cfg.model_type == "sheaf" else t.mpnn_train_rounds
    k_eval = t.K_eval if cfg.model_type == "sheaf" else t.mpnn_eval_rounds
    return model, task, cfg, int(trained), int(k_eval)


@torch.no_grad()
def final_logits(model, model_type: str, fwd: dict, k: int) -> torch.Tensor:
    if model_type == "sheaf":
        window, _, _ = model(fwd["patches"], fwd["edge_indices"], num_iters=k, loss_window=1,
                             **fwd["model_kwargs"], training=False)
        return window[-1]
    logits, _ = model(fwd["patches"], fwd["edge_indices"], num_rounds=k,
                      **fwd["model_kwargs"], training=False)
    return logits


def per_board(model, task, model_type, data_dir, split, ks, batch_size, device, max_boards=None):
    """``solved [len(ks), M]`` bool and ``line_ok [len(ks), M]`` (fraction of the 2n clues),
    for the first ``M`` boards of the split (all, or ``max_boards``)."""
    ds = PuzzleDataset(data_dir, split)
    solved, line_ok = [[] for _ in ks], [[] for _ in ks]
    seen = 0
    for _, batch in ds.iter_test_batches(batch_size):
        if max_boards is not None and seen >= max_boards:
            break
        seen += len(batch["inputs"])
        fwd, targets, _ = task.prepare(batch)
        fwd, targets = move_to_device(fwd, device), move_to_device(targets, device)
        for i, k in enumerate(ks):
            views = final_logits(model, model_type, fwd, k).permute(1, 0, 2, 3)  # [B, 2n, n, 2]
            pred = V.reassemble_picross_logits(views).argmax(-1)
            ok = picross_line_clues(V.picross_slice_batch(pred[..., None])[..., 0])
            ok = (ok == targets["clues"]).all(-1)
            solved[i].append((pred == targets["labels_grid"]).all(dim=(1, 2)).cpu())
            line_ok[i].append(ok.float().mean(-1).cpu())
    m = max_boards if max_boards is not None else seen
    return (torch.stack([torch.cat(s) for s in solved]).numpy()[:, :m],
            torch.stack([torch.cat(s) for s in line_ok]).numpy()[:, :m])


def compare_k(spec: str, info: dict) -> int:
    if spec == "trained":
        return int(info["trained_k"])
    if spec == "eval":
        return int(info["eval_k"])
    return int(spec)


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 99 else f"{lo}+")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--iters", type=int, nargs="+",
                   default=[1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100])
    p.add_argument("--splits", nargs="+", default=["test_easy", "test_hard"])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--params", choices=("ema", "live"), default="ema")
    p.add_argument("--max-boards", type=int, default=None, help="first N boards per split")
    p.add_argument("--compare-k", default="trained",
                   help="K for solved_vs_rounds: 'trained', 'eval' or an integer")
    p.add_argument("--reuse", action="store_true",
                   help="reload per_board.npz / results.json from --out-dir (no model runs)")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/analysis_picross"))
    a = p.parse_args()
    labels = a.labels or [c.parent.name for c in a.checkpoints]
    if len(labels) != len(a.checkpoints):
        raise SystemExit("--labels must match the number of checkpoints")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    ks = sorted(set(a.iters))

    results, raw, meta = {}, {}, {}
    data_dir = None
    if a.reuse:
        saved = json.loads((a.out_dir / "results.json").read_text())
        meta, results = saved["meta"], saved["solved_vs_k"]
        with np.load(a.out_dir / "per_board.npz") as npz:
            raw = {k: npz[k] for k in npz.files}
        missing = set(labels) - set(meta)
        if missing:
            raise SystemExit(f"--reuse: no saved results for {sorted(missing)}")
        ckpt = torch.load(Path(meta[labels[0]]["checkpoint"]), map_location="cpu",
                          weights_only=False)
        data_dir = Path(ckpt["config"]["data"]["dir"])
    for path, label in zip([] if a.reuse else a.checkpoints, labels, strict=False):
        model, task, cfg, trained, k_eval = load_model(path, a.params, device)
        ks_model = sorted(set(ks) | {k_eval})
        data_dir = Path(cfg.data.dir)
        meta[label] = {"model_type": cfg.model_type, "trained_k": trained, "eval_k": k_eval,
                       "lr": float(cfg.training.lr), "checkpoint": str(path)}
        print(f"[{label}] {cfg.model_type} lr={cfg.training.lr} trained K={trained} "
              f"eval K={k_eval}")
        for split in a.splits:
            s, lo = per_board(model, task, cfg.model_type, data_dir, split, ks_model,
                              a.batch_size, device, a.max_boards)
            raw[f"{label}/{split}/solved"], raw[f"{label}/{split}/line_ok"] = s, lo
            raw[f"{label}/{split}/ks"] = np.array(ks_model)
            results.setdefault(label, {})[split] = {
                str(k): {"solved": float(s[i].mean()), "line_ok": float(lo[i].mean())}
                for i, k in enumerate(ks_model)
            }
            print(f"  {split:10s} " + "  ".join(
                f"K={k}:{100 * s[i].mean():.1f}" for i, k in enumerate(ks_model)))
        del model
        torch.cuda.empty_cache()

    rounds = {s: np.load(data_dir / s / "all__rounds.npy")[: a.max_boards] for s in a.splits}

    # --- solved vs K -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(a.splits), figsize=(6 * len(a.splits), 4), squeeze=False)
    for ax, split in zip(axes[0], a.splits, strict=True):
        for label in labels:
            kk = raw[f"{label}/{split}/ks"]
            line, = ax.plot(kk, 100 * raw[f"{label}/{split}/solved"].mean(1), "o-", label=label)
            ax.axvline(meta[label]["trained_k"], color=line.get_color(), ls=":", lw=1)
        ax.set_xscale("log")
        ax.set_xlabel("K (ADMM iterations / MPNN rounds) at eval; dotted = trained K")
        ax.set_ylabel("solved boards (%)")
        ax.set_title(split)
        ax.grid(alpha=0.3)
        ax.legend()
    save_figure(fig, str(a.out_dir / "solved_vs_k.png"))

    # --- solved vs line-logic rounds, at each model's eval K -----------------------------------
    by_rounds = {}
    fig, axes = plt.subplots(1, len(a.splits), figsize=(6 * len(a.splits), 4), squeeze=False)
    for ax, split in zip(axes[0], a.splits, strict=True):
        r = rounds[split]
        bins = [(lo, hi) for lo, hi in ROUND_BINS if ((r >= lo) & (r <= hi)).any()]
        x = np.arange(len(bins))
        width = 0.8 / len(labels)
        for j, label in enumerate(labels):
            kk = list(raw[f"{label}/{split}/ks"])
            k_cmp = compare_k(a.compare_k, meta[label])
            s = raw[f"{label}/{split}/solved"][kk.index(k_cmp)]
            rates = [100 * s[(r >= lo) & (r <= hi)].mean() for lo, hi in bins]
            by_rounds.setdefault(label, {})[split] = {
                bin_label(lo, hi): {"n": int(((r >= lo) & (r <= hi)).sum()), "solved": rate / 100}
                for (lo, hi), rate in zip(bins, rates, strict=True)
            }
            ax.bar(x + j * width, rates, width, label=f"{label} (K={k_cmp})")
        counts = [int(((r >= lo) & (r <= hi)).sum()) for lo, hi in bins]
        ax.set_xticks(x + width * (len(labels) - 1) / 2,
                      [f"{bin_label(lo, hi)}\n(n={c})" for (lo, hi), c in zip(bins, counts,
                                                                             strict=True)])
        ax.set_xlabel("parallel line-logic rounds of the board")
        ax.set_ylabel("solved boards (%)")
        ax.set_title(split)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
    save_figure(fig, str(a.out_dir / "solved_vs_rounds.png"))

    # --- first K that solves a board vs its line-logic rounds (test_easy) ----------------------
    first_k = {}
    split = a.splits[0]
    r = rounds[split]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label in labels:
        kk = np.array(raw[f"{label}/{split}/ks"])
        s = raw[f"{label}/{split}/solved"]
        any_solved = s.any(0)
        first = np.where(any_solved, kk[np.argmax(s, axis=0)], np.inf)
        stats = {}
        for v in np.unique(r):
            sel = (r == v) & any_solved
            if sel.sum() >= 5:
                med = float(np.median(first[sel]))
                stats[int(v)] = {"n": int(sel.sum()), "median_first_k": med}
        first_k[label] = stats
        xs = sorted(stats)
        ax.plot(xs, [stats[v]["median_first_k"] for v in xs], "o-", label=label)
        # forgetting: solved at some K but not at the largest K
        lost = any_solved & ~s[-1]
        print(f"[{label}] {split}: solved at some K = {100 * any_solved.mean():.1f} %, "
              f"of which not at K={kk[-1]}: {100 * lost.sum() / max(any_solved.sum(), 1):.1f} %")
    xs_all = np.arange(1, int(r.max()) + 1)
    ax.plot(xs_all, xs_all, color="grey", ls="--", lw=1, label="K = rounds")
    ax.set_yscale("log")
    ax.set_xlabel("parallel line-logic rounds of the board")
    ax.set_ylabel("median smallest K that solves it")
    ax.set_title(f"{split}: iterations needed vs propagation depth (boards solved at some K)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, str(a.out_dir / "first_k_vs_rounds.png"))

    (a.out_dir / "results.json").write_text(json.dumps(
        {"meta": meta, "solved_vs_k": results, "solved_vs_rounds": by_rounds,
         "first_k_vs_rounds": first_k}, indent=2))
    np.savez_compressed(a.out_dir / "per_board.npz", **raw)
    print(f"wrote {a.out_dir}/")


if __name__ == "__main__":
    main()
