"""Picross primitives (clues, packing, line solver, solution counting) and the offline builder."""

from __future__ import annotations

import json
from math import comb

import numpy as np
import pytest

from sheaf_admm.data import picross as P
from sheaf_admm.data.build_picross import PicrossConfig, build_from_arrays
from sheaf_admm.data.loaders import PuzzleDataset


def _brute_force_placements(clue, n):
    lines = ((np.arange(1 << n)[:, None] >> np.arange(n)[::-1]) & 1).astype(np.int8)
    return {tuple(line) for line in lines if P.line_clue(line) == clue}


@pytest.mark.parametrize("clue", [(), (5,), (1, 1), (2, 1), (1, 1, 1), (3,)])
def test_line_placements_match_brute_force(clue):
    n = 5
    got = P.line_placements(clue, n)
    assert {tuple(r) for r in got} == _brute_force_placements(clue, n)
    if clue:
        free = n - (sum(clue) + len(clue) - 1)
        assert len(got) == comb(free + len(clue), len(clue))


def test_line_placements_rejects_overfull_clue():
    with pytest.raises(ValueError):
        P.line_placements((3, 3), 6)


def test_clues_pack_unpack_roundtrip():
    rng = np.random.default_rng(0)
    for n in (4, 5, 10):
        for _ in range(20):
            grid = (rng.random((n, n)) < 0.55).astype(np.uint8)
            clues = P.grid_clues(grid)
            packed = P.pack_clues(clues, n)
            assert packed.shape == (2 * n, P.clue_slots(n))
            assert P.unpack_clues(packed) == clues
            assert P.satisfies(grid, clues)


def test_packed_layout_is_right_aligned_rows_then_columns():
    grid = np.array([[1, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]], dtype=np.uint8)
    packed = P.pack_clues(P.grid_clues(grid), 4)
    np.testing.assert_array_equal(
        packed,
        [[1, 2], [0, 0], [0, 4], [0, 1], [1, 1], [0, 2], [1, 1], [1, 1]],
    )


def test_solve_unique_line_solvable():
    grid = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)  # plus sign
    res = P.solve(P.grid_clues(grid), 3)
    assert res.num_solutions == 1 and res.line_solvable
    np.testing.assert_array_equal(res.solution, grid)
    assert res.rounds >= 1


def test_solve_detects_multiple_solutions():
    diagonal = np.eye(2, dtype=np.uint8)  # its anti-diagonal has the same clues
    res = P.solve(P.grid_clues(diagonal), 2)
    assert res.num_solutions == 2 and not res.line_solvable and res.solution is None


def test_solve_detects_unsatisfiable_clues():
    clues = [(2,), (), (), ()]  # rows need 2 cells, columns need none
    assert P.solve(clues, 2).num_solutions == 0


def test_solve_counts_match_brute_force_on_small_boards():
    """On random 4x4 boards, the solution count agrees with exhaustive enumeration, and
    search-free (line-solvable) boards are always unique."""
    n = 4
    boards = ((np.arange(1 << (n * n))[:, None] >> np.arange(n * n)) & 1).reshape(-1, n, n)
    keys = [tuple(map(tuple, P.grid_clues(b))) for b in boards]
    counts: dict = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    rng = np.random.default_rng(1)
    for i in rng.choice(len(boards), size=200, replace=False):
        res = P.solve(P.grid_clues(boards[i]), n, limit=2)
        assert res.num_solutions == min(counts[keys[i]], 2)
        if res.line_solvable:
            assert res.num_solutions == 1


def test_parallel_rounds_at_least_sequential():
    rng = np.random.default_rng(2)
    n = 10
    for _ in range(20):
        grid = (rng.random((n, n)) < 0.6).astype(np.uint8)
        cands = [P.line_placements(c, n) for c in P.grid_clues(grid)]
        empty = np.full((n, n), P.UNKNOWN, dtype=np.int8)
        seq = P.propagate(empty.copy(), [c.copy() for c in cands])
        par = P.propagate(empty.copy(), [c.copy() for c in cands], parallel=True)
        np.testing.assert_array_equal(seq[0], par[0])  # same fixed point
        assert par[2] >= seq[2]


def test_build_from_arrays_writes_clean_disjoint_splits(tmp_path):
    n = 8  # random 8x8 boards at density 0.5 give a few unique-but-search puzzles (test_hard)
    rng = np.random.default_rng(3)
    y = (rng.random((500, n * n)) < 0.5).astype(np.uint8)
    x = np.stack([P.pack_clues(P.grid_clues(g.reshape(n, n)), n).reshape(-1) for g in y])
    x_tr, y_tr = np.concatenate([x[:200], x[:10]]), np.concatenate([y[:200], y[:10]])
    x_te, y_te = np.concatenate([x[200:], x[:5]]), np.concatenate([y[200:], y[:5]])

    cfg = PicrossConfig(size=n, output_dir=tmp_path, workers=1)
    stats = build_from_arrays(cfg, x_tr, y_tr, x_te, y_te)
    assert stats["train_duplicates"] == 10
    assert stats["test_in_train"] == 5
    assert stats["test_multi_solution"] > 0
    assert stats["test_hard"]["size"] > 0 and stats["test_easy"]["size"] > 0

    seen = {}
    for split in ("train", "test_easy", "test_hard"):
        ds = PuzzleDataset(tmp_path, split)
        s = ds.sets["all"]
        assert s.inputs.shape[1] == n * n and s.labels.shape[1] == n * n
        assert ds.metadata.height == n and ds.metadata.width == n
        assert len(s.inputs) == stats[split]["size"]
        for xi, yi in zip(s.inputs, s.labels, strict=True):
            res = P.solve(P.unpack_clues(xi.reshape(2 * n, n // 2)), n)
            assert res.num_solutions == 1
            np.testing.assert_array_equal(res.solution.reshape(-1), yi)
        seen[split] = {r.tobytes() for r in s.inputs}
        solvable = np.load(tmp_path / split / "all__line_solvable.npy")
        if split == "test_easy":
            assert solvable.all()
        elif split == "test_hard":
            assert not solvable.any()
    assert not (seen["train"] & (seen["test_easy"] | seen["test_hard"]))
    assert json.loads((tmp_path / "stats.json").read_text())["train"]["size"] == len(seen["train"])


def test_build_rejects_inconsistent_source(tmp_path):
    n = 10
    y = np.zeros((2, n * n), dtype=np.uint8)
    x = np.zeros((2, n * n), dtype=np.uint8)
    x[0, 4] = 3  # claims a run of 3 on an empty row
    with pytest.raises(ValueError, match="inconsistent"):
        build_from_arrays(PicrossConfig(output_dir=tmp_path, workers=1), x, y, x[1:], y[1:])
