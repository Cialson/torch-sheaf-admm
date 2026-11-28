"""Learning-rate sweep for the Picross models, scored on the validation splits.

For every (experiment, lr) pair, runs ``scripts/train.py`` for a few epochs and evaluates on
``val_easy`` / ``val_hard`` (held out of the train boards by ``build_picross``; the test splits
are never used for selection) at the trained number of iterations / rounds. Finished runs are
skipped, so the sweep can be interrupted and resumed. Writes ``summary.md`` / ``summary.json``.

Sweep-specific overrides (same for every model, so the comparison stays fair):

* ``K_eval`` = ``mpnn_eval_rounds`` = ``--k`` (default 20, the trained horizon): the MPNN degrades
  beyond its trained rounds, so evaluating at 50 would penalize it;
* ``ema_decay=0.995``: with the default 0.999, one epoch (~2.1k steps) leaves ~12 % of the random
  init in the EMA weights used at eval.

    uv run python scripts/sweep_lr.py                       # 3 models x 4 lr, 1 epoch (~1h30)
    uv run python scripts/sweep_lr.py --summary-only        # rebuild the table
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("val_easy", "val_hard")


def run_dir(out: Path, exp: str, lr: float) -> Path:
    return out / exp / f"lr{lr:g}"


def train(exp: str, lr: float, args) -> None:
    d = run_dir(args.out_dir, exp, lr)
    if (d / "history.json").exists():
        print(f"[skip] {exp} lr={lr:g} (done)")
        return
    d.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "scripts/train.py", f"+experiment={exp}",
        f"training.lr={lr}", f"training.epochs={args.epochs}", "training.val_interval=1",
        "training.ema_decay=0.995", f"training.K_eval={args.k}",
        f"training.mpnn_eval_rounds={args.k}", f"training.seed={args.seed}",
        f"data.val_splits=[{','.join(SPLITS)}]", f"hydra.run.dir={d}", "wandb.mode=disabled",
    ]
    print(f"[run] {exp} lr={lr:g} -> {d}", flush=True)
    t0 = time.time()
    with (d / "train.log").open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    status = "ok" if proc.returncode == 0 else f"exit {proc.returncode}"
    (d / "sweep_status.json").write_text(
        json.dumps({"status": status, "minutes": (time.time() - t0) / 60}))
    print(f"[end] {exp} lr={lr:g}: {status} in {(time.time() - t0) / 60:.1f} min", flush=True)


def summarize(args) -> None:
    rows = {}
    for exp in args.experiments:  # every lr run found on disk, not only this invocation's --lrs
        found = {float(d.name[2:]) for d in (args.out_dir / exp).glob("lr*") if d.is_dir()}
        for lr in sorted(found | set(args.lrs)):
            d = run_dir(args.out_dir, exp, lr)
            hist = d / "history.json"
            if not hist.exists():
                rows.setdefault(exp, []).append({"lr": lr, "status": "missing / diverged"})
                continue
            last = json.loads(hist.read_text())[-1]
            row = {"lr": lr, "status": "ok", "loss": last.get("loss")}
            for s in SPLITS:
                m = last.get(s, {})
                row[f"{s}_solved"] = m.get("solved")
                row[f"{s}_line_ok"] = m.get("line_ok")
            st = d / "sweep_status.json"
            if st.exists():
                row["minutes"] = json.loads(st.read_text())["minutes"]
            rows.setdefault(exp, []).append(row)

    def key(r):  # select on val_hard solved, tie-break on val_easy solved
        return (r.get("val_hard_solved") or -1, r.get("val_easy_solved") or -1)

    lines = [f"# LR sweep ({args.epochs} epoch(s), eval K={args.k}, seed {args.seed})", "",
             "| experiment | lr | val_easy solved | val_hard solved | val_hard line_ok | "
             "last-batch loss | min |", "|---|---|---|---|---|---|---|"]
    best = {}
    for exp, rs in rows.items():
        ok = [r for r in rs if r["status"] == "ok" and r.get("val_hard_solved") is not None]
        if ok:
            best[exp] = max(ok, key=key)["lr"]
        for r in rs:
            if r["status"] != "ok":
                lines.append(f"| {exp} | {r['lr']:g} | {r['status']} | | | | |")
                continue
            mark = " **(best)**" if best.get(exp) == r["lr"] else ""
            pct = lambda v: "-" if v is None else f"{100 * v:.1f}"  # noqa: E731
            lines.append(
                f"| {exp} | {r['lr']:g}{mark} | {pct(r['val_easy_solved'])} | "
                f"{pct(r['val_hard_solved'])} | {pct(r['val_hard_line_ok'])} | "
                f"{r['loss']:.4f} | {r.get('minutes', float('nan')):.1f} |")
    text = "\n".join(lines) + "\n"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.md").write_text(text)
    (args.out_dir / "summary.json").write_text(json.dumps({"rows": rows, "best": best}, indent=2))
    print(text)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--experiments", nargs="+",
                   default=["picross_mpnn", "picross_mpnn_comm", "picross_sheaf"])
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 1.7e-3, 3e-3])
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--k", type=int, default=20, help="eval iterations / rounds (trained K)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/lr_sweep"))
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()
    if not args.summary_only:
        for exp in args.experiments:
            for lr in args.lrs:
                train(exp, lr, args)
    summarize(args)


if __name__ == "__main__":
    main()
