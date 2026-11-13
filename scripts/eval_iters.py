"""Evaluate a trained checkpoint with several ADMM iteration / MPNN round counts.

Rebuilds the model and task from the config stored in ``checkpoint.pt``, loads the EMA
weights (as ``train.py`` does at eval), and reports the task metrics for each count:

    python scripts/eval_iters.py outputs/<date>/<time>/checkpoint.pt --iters 5 10 20 50 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sheaf_admm.data import PuzzleDataset  # noqa: E402
from sheaf_admm.models import model_config_from_dict  # noqa: E402
from sheaf_admm.training import build_model, create_train_state, evaluate, make_task  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--iters", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    p.add_argument("--splits", nargs="+", default=None, help="default: the run's val_splits")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--params", choices=("ema", "live"), default="ema")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    if cfg.data.loader != "puzzle":
        raise SystemExit("only puzzle datasets (maze / sudoku / life / picross) are supported")
    task = make_task(cfg.task, **OmegaConf.to_container(cfg.task_cfg, resolve=True))
    model_cfg = model_config_from_dict(OmegaConf.to_container(cfg.model, resolve=True))
    model = build_model(model_cfg, cfg.model_type, seed=0)
    splits = args.splits or list(cfg.data.val_splits)

    _, sample = next(PuzzleDataset(cfg.data.dir, splits[0]).iter_test_batches(args.batch_size))
    sample_fwd, _, _ = task.prepare(sample)
    t = cfg.training
    state = create_train_state(
        model, sample_fwd, model_type=cfg.model_type, lr=t.lr, weight_decay=t.weight_decay,
        warmup_steps=t.warmup_steps, grad_clip=t.grad_clip, ema_decay=t.ema_decay,
        k_init=t.K_train, loss_window=t.loss_window, seed=0,
    )
    device = next(state.model.parameters()).device
    state.model.load_state_dict(ckpt["model_state"])
    use_ema = args.params == "ema" and ckpt.get("ema_state") is not None
    state.ema_params = (
        {k: v.to(device) for k, v in ckpt["ema_state"].items()} if use_ema else None
    )
    trained = t.K_train if cfg.model_type == "sheaf" else t.mpnn_train_rounds
    print(f"{cfg.model_type}  step={ckpt['step']}  params={'ema' if use_ema else 'live'}  "
          f"iterations at train time={trained}")

    for split in splits:
        ds = PuzzleDataset(cfg.data.dir, split)
        for k in args.iters:
            m = evaluate(
                state, task, (b for _, b in ds.iter_test_batches(args.batch_size)),
                model_type=cfg.model_type, graph_readout=model_cfg.mpnn_graph_readout, k_eval=k,
            )
            print(f"  {split:10s} K={k:4d}  " + "  ".join(f"{n}={v * 100:6.2f}%" for n, v in m.items()))


if __name__ == "__main__":
    main()
