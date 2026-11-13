"""Game of Life dataset in the maze puzzle format (dead=2, alive=5; MazeTask pads with wall=1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sheaf_admm.data.common import PuzzleDatasetMetadata, save_npy

DEAD, ALIVE = 2, 5


def life_step(g: np.ndarray) -> np.ndarray:
    """One generation for a batch ``[B, H, W]`` of bool grids; cells outside are dead."""
    h, w = g.shape[1:]
    p = np.pad(g, ((0, 0), (1, 1), (1, 1))).astype(np.int8)
    n = sum(
        p[:, 1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if dy or dx
    )
    return (n == 3) | (g & (n == 2))


def write_split(out: Path, n: int, size: int, steps: int, density: float, rng) -> None:
    x = rng.random((n, size, size)) < density
    y = x
    for _ in range(steps):
        y = life_step(y)
    tokens = lambda a: np.where(a, ALIVE, DEAD).reshape(n, -1).astype(np.int32)  # noqa: E731
    save_npy(out / "all__inputs.npy", tokens(x))
    save_npy(out / "all__labels.npy", tokens(y))
    idx = np.arange(n + 1, dtype=np.int32)  # one example per puzzle, one puzzle per group
    save_npy(out / "all__puzzle_indices.npy", idx)
    save_npy(out / "all__group_indices.npy", idx)
    save_npy(out / "all__puzzle_identifiers.npy", np.zeros(n, dtype=np.int32))
    PuzzleDatasetMetadata(
        pad_id=0, ignore_label_id=0, blank_identifier_id=0, vocab_size=6,
        seq_len=size * size, num_puzzle_identifiers=1, total_groups=n,
        mean_puzzle_examples=1.0, total_puzzles=n, sets=["all"], height=size, width=size,
    ).write(out / "dataset.json")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("datasets/life_15px_1step"))
    p.add_argument("--size", type=int, default=15)        # odd: stride-2 agents cover the grid
    p.add_argument("--steps", type=int, default=1)        # generations to predict
    p.add_argument("--density", type=float, default=0.3)
    p.add_argument("--train-size", type=int, default=10000)
    p.add_argument("--test-size", type=int, default=1000)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    write_split(a.output_dir / "train", a.train_size, a.size, a.steps, a.density, rng)
    write_split(a.output_dir / "test", a.test_size, a.size, a.steps, a.density, rng)
    write_split(a.output_dir / "test_2x", a.test_size, 2 * a.size + 1, a.steps, a.density, rng)


if __name__ == "__main__":
    main()
