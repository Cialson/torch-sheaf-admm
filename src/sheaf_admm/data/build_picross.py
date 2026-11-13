"""Picross builder: download NonoDataset 10x10, keep unique-solution boards, split by difficulty.

Source: NonoDataset (Buades Rubio, Moya-Alcover, Jaume-i-Capó, López González, "Solving
nonograms using neural networks", Entertainment Computing 50, 2024), CC BY 4.0,
https://github.com/josebambu/NonoDataset (pinned commit, files checked by SHA-256),
also on Zenodo (doi:10.5281/zenodo.14620418). Its 10x10 boards come from icon images (train and
test) and from noise / random shapes (train only). About a quarter of them have several
solutions, and some test boards also appear in train, so the builder:

1. downloads the four 10x10 ``.npz`` files (~75 MB) into ``--raw-dir``
   (default ``<output-dir>/raw``) once, and loads them with ``allow_pickle=False``;
2. recomputes the clues from each solution and checks they match the source clues;
3. removes duplicate clue sets inside train, and test boards whose clues appear in train;
4. counts solutions (up to 2) with :mod:`sheaf_admm.data.picross` and drops non-unique boards;
5. writes ``train`` (all unique source-train boards), ``test_easy`` (unique source-test boards
   solved by line logic alone) and ``test_hard`` (unique, but needing search).

On-disk format is the puzzle layout (``seq_len = 100``): inputs are the packed clues
``[20 lines (10 rows, then 10 columns), 5 slots]`` right-aligned, values 0-10 (0 = padding);
labels are the 0/1 solution grid, row-major. Each split also stores ``all__rounds.npy``
(parallel line-logic rounds to the fixed point, a difficulty proxy to compare with the number of
ADMM iterations) and ``all__line_solvable.npy``; the puzzle loader ignores both.

Usage:
    python -m sheaf_admm.data.build_picross --output-dir datasets/picross_10x10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from dataclasses import asdict, dataclass
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .common import PuzzleDatasetMetadata, save_npy
from .picross import clue_slots, grid_clues, pack_clues, solve, unpack_clues

SOURCE_COMMIT = "1ab83daabc7ff116fdf917de1c9057dd2aca82b5"
SOURCE_URL = f"https://raw.githubusercontent.com/josebambu/NonoDataset/{SOURCE_COMMIT}/10x10/"
SOURCE_FILES = {
    "x_train_dataset.npz": "c8d4571c6fcdefc14e882acbd820cc656dd7f7e5526f7ab9cdffcf458f971490",
    "y_train_dataset.npz": "85db407c693cf429b9fbf0d667e15188e5bfda8ae497110291ac25b4779a67cf",
    "x_test_dataset.npz": "25fddf5d96f738afbae46854a863d88ca6a62c24c8c6ec5848ab94041d72a56e",
    "y_test_dataset.npz": "a20b56d7058cc7db42cf262a0bf324c23598040636fc5abbb25e8e9539728e55",
}


@dataclass
class PicrossConfig:
    size: int = 10
    output_dir: Path = Path("datasets/picross_10x10")
    raw_dir: Path | None = None  # default: <output_dir>/raw
    train_limit: int | None = None  # cap source-train boards (debugging)
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    seed: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["raw_dir"] = str(self.resolved_raw_dir)
        return payload

    @property
    def resolved_raw_dir(self) -> Path:
        return self.raw_dir if self.raw_dir is not None else self.output_dir / "raw"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_source(raw_dir: Path) -> None:
    """Fetch the pinned NonoDataset 10x10 files into ``raw_dir`` (skipped if already valid)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in SOURCE_FILES.items():
        path = raw_dir / name
        if path.exists() and _sha256(path) == digest:
            continue
        print(f"[download] {SOURCE_URL}{name}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(SOURCE_URL + name, tmp)  # noqa: S310 (fixed https URL)
        if _sha256(tmp) != digest:
            tmp.unlink()
            raise RuntimeError(f"checksum mismatch for {name}; the source may have changed")
        tmp.replace(path)


def _load_npz(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        return data[data.files[0]]


def canonical_inputs(labels: np.ndarray, n: int) -> np.ndarray:
    """Packed clues ``[M, 2n * ceil(n/2)]`` uint8 recomputed from 0/1 solutions ``[M, n*n]``."""
    grids = labels.reshape(-1, n, n)
    if len(grids) == 0:
        return np.zeros((0, 2 * n * clue_slots(n)), dtype=np.uint8)
    return np.stack([pack_clues(grid_clues(g), n).reshape(-1) for g in grids])


def _analyse(packed: np.ndarray, n: int) -> tuple[int, bool, int]:
    res = solve(unpack_clues(packed.reshape(2 * n, clue_slots(n))), n, limit=2)
    return res.num_solutions, res.line_solvable, res.rounds


def analyse_all(inputs: np.ndarray, n: int, workers: int) -> dict[str, np.ndarray]:
    fn = partial(_analyse, n=n)
    if workers > 1 and len(inputs) > 1000:
        with Pool(workers) as pool:
            rows = pool.map(fn, list(inputs), chunksize=512)
    else:
        rows = [fn(x) for x in inputs]
    out = np.array(rows, dtype=np.int64).reshape(-1, 3)
    return {
        "num_solutions": out[:, 0],
        "line_solvable": out[:, 1].astype(bool),
        "rounds": out[:, 2].astype(np.int32),
    }


def _first_occurrence(inputs: np.ndarray) -> np.ndarray:
    """Mask keeping the first row of each distinct clue set."""
    _, first = np.unique(inputs, axis=0, return_index=True)
    mask = np.zeros(len(inputs), dtype=bool)
    mask[first] = True
    return mask


def write_split(
    out: Path, inputs: np.ndarray, labels: np.ndarray, rounds: np.ndarray, line_solvable, n: int
) -> None:
    m = len(inputs)
    save_npy(out / "all__inputs.npy", inputs.astype(np.uint8))
    save_npy(out / "all__labels.npy", labels.astype(np.uint8))
    idx = np.arange(m + 1, dtype=np.int32)  # one example per puzzle, one puzzle per group
    save_npy(out / "all__puzzle_indices.npy", idx)
    save_npy(out / "all__group_indices.npy", idx)
    save_npy(out / "all__puzzle_identifiers.npy", np.zeros(m, dtype=np.int32))
    save_npy(out / "all__rounds.npy", rounds.astype(np.int32))
    save_npy(out / "all__line_solvable.npy", np.asarray(line_solvable, dtype=bool))
    PuzzleDatasetMetadata(
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        vocab_size=n + 1,  # clue values 0..n (labels are 0/1)
        seq_len=n * n,
        num_puzzle_identifiers=1,
        total_groups=m,
        mean_puzzle_examples=1.0,
        total_puzzles=m,
        sets=["all"],
        height=n,
        width=n,
    ).write(out / "dataset.json")


def build_from_arrays(
    cfg: PicrossConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Filter, deduplicate and write the splits from source arrays; returns build statistics."""
    n = cfg.size
    if n % 2:
        raise ValueError("odd sizes do not pack into n*n inputs; use an even size")
    stats: dict = {}
    sources = {"train": (x_train, y_train), "test": (x_test, y_test)}
    clean = {}
    for name, (x, y) in sources.items():
        y = (np.asarray(y).reshape(len(y), n * n) > 0).astype(np.uint8)
        x_canon = canonical_inputs(y, n)
        x_src = np.asarray(x).reshape(len(x), x_canon.shape[1])
        mismatch = int((x_canon != x_src).any(axis=1).sum())
        if mismatch:
            raise ValueError(f"{mismatch} {name} boards have clues inconsistent with their labels")
        clean[name] = (x_canon, y)
        stats[f"source_{name}"] = len(y)

    x_tr, y_tr = clean["train"]
    if cfg.train_limit is not None:
        x_tr, y_tr = x_tr[: cfg.train_limit], y_tr[: cfg.train_limit]
    keep = _first_occurrence(x_tr)
    stats["train_duplicates"] = int((~keep).sum())
    x_tr, y_tr = x_tr[keep], y_tr[keep]

    x_te, y_te = clean["test"]
    keep = _first_occurrence(x_te)
    train_keys = {row.tobytes() for row in x_tr}
    in_train = np.array([row.tobytes() in train_keys for row in x_te], dtype=bool)
    stats["test_duplicates"] = int((~keep).sum())
    stats["test_in_train"] = int((keep & in_train).sum())
    keep &= ~in_train
    x_te, y_te = x_te[keep], y_te[keep]

    rng = np.random.default_rng(cfg.seed)
    for name, x, y in (("train", x_tr, y_tr), ("test", x_te, y_te)):
        print(f"[analyse] {name}: {len(x)} boards on {cfg.workers} workers")
        a = analyse_all(x, n, cfg.workers)
        unique = a["num_solutions"] == 1
        stats[f"{name}_multi_solution"] = int((a["num_solutions"] >= 2).sum())
        stats[f"{name}_unsat"] = int((a["num_solutions"] == 0).sum())
        if name == "train":
            splits = {"train": unique}
        else:
            splits = {
                "test_easy": unique & a["line_solvable"],
                "test_hard": unique & ~a["line_solvable"],
            }
        for split, mask in splits.items():
            idx = np.flatnonzero(mask)
            if len(idx) == 0:
                print(f"[warn] {split} is empty; not written")
                stats[split] = {"size": 0}
                continue
            if split == "train":
                idx = rng.permutation(idx)
            write_split(
                cfg.output_dir / split,
                x[idx],
                y[idx],
                a["rounds"][idx],
                a["line_solvable"][idx],
                n,
            )
            stats[split] = {
                "size": len(idx),
                "line_solvable": float(a["line_solvable"][idx].mean()),
                "rounds_p50_p90_max": [
                    int(v) for v in np.percentile(a["rounds"][idx], [50, 90, 100])
                ],
                "fill_density": float(y[idx].mean()),
            }
    (cfg.output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    (cfg.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    return stats


def build(cfg: PicrossConfig) -> dict:
    if cfg.size != 10:
        raise ValueError("only the NonoDataset 10x10 files are wired in")
    raw = cfg.resolved_raw_dir
    download_source(raw)
    arrays = [_load_npz(raw / name) for name in SOURCE_FILES]
    stats = build_from_arrays(cfg, *arrays)
    print(json.dumps(stats, indent=2))
    print(f"Done. Dataset saved to {cfg.output_dir}")
    return stats


def parse_args() -> PicrossConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=PicrossConfig.output_dir)
    p.add_argument("--raw-dir", type=Path, default=None)
    p.add_argument("--train-limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=PicrossConfig.workers)
    p.add_argument("--seed", type=int, default=PicrossConfig.seed)
    a = p.parse_args()
    return PicrossConfig(
        output_dir=a.output_dir,
        raw_dir=a.raw_dir,
        train_limit=a.train_limit,
        workers=a.workers,
        seed=a.seed,
    )


if __name__ == "__main__":
    build(parse_args())
