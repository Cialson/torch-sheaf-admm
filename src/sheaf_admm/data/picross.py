"""Nonogram (Picross) primitives: clues, packed layout, line solver, solution counting.

A board is an ``n x n`` 0/1 grid. Its clues are ``2n`` lines (the ``n`` rows top to bottom,
then the ``n`` columns left to right), each the tuple of run lengths of filled cells.

Packed layout (the dataset input format, same as NonoDataset): ``[2n, L]`` with
``L = ceil(n / 2)`` slots per line, runs right-aligned and left-padded with 0. For even ``n``
this is exactly ``n * n`` values, so a puzzle's inputs and labels share ``seq_len = n * n``.

The solver is plain NumPy and only meant for dataset building and evaluation:

* :func:`propagate` applies line logic (each line keeps the placements consistent with the known
  cells, and cells common to all of them are fixed) until nothing changes, either sequentially
  (rows then columns, as a human would) or in parallel rounds (every line deduces from the same
  snapshot, like one synchronous message-passing / ADMM round).
* :func:`solve` adds backtracking to count solutions up to a limit, and reports whether line
  logic alone suffices.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import combinations

import numpy as np

UNKNOWN = -1

Clue = tuple[int, ...]


def clue_slots(n: int) -> int:
    """Number of clue slots per line: a line of ``n`` cells has at most ``ceil(n/2)`` runs."""
    return (n + 1) // 2


def line_clue(line: np.ndarray) -> Clue:
    """Run lengths of the filled cells of a 0/1 line."""
    runs, count = [], 0
    for v in np.asarray(line).tolist():
        if v:
            count += 1
        elif count:
            runs.append(count)
            count = 0
    if count:
        runs.append(count)
    return tuple(runs)


def grid_clues(grid: np.ndarray) -> list[Clue]:
    """The ``2n`` clues of a square 0/1 grid: rows first, then columns."""
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] != g.shape[1]:
        raise ValueError(f"expected a square grid, got shape {g.shape}")
    return [line_clue(r) for r in g] + [line_clue(c) for c in g.T]


def pack_clues(clues: list[Clue], n: int) -> np.ndarray:
    """``2n`` clues -> ``[2n, ceil(n/2)]`` uint8, runs right-aligned, left-padded with 0."""
    slots = clue_slots(n)
    if len(clues) != 2 * n:
        raise ValueError(f"expected {2 * n} clues, got {len(clues)}")
    out = np.zeros((2 * n, slots), dtype=np.uint8)
    for k, clue in enumerate(clues):
        if len(clue) > slots:
            raise ValueError(f"clue {clue} has more than {slots} runs")
        if clue:
            out[k, slots - len(clue) :] = clue
    return out


def unpack_clues(packed: np.ndarray) -> list[Clue]:
    """Inverse of :func:`pack_clues` (zeros are padding)."""
    return [tuple(int(v) for v in row if v) for row in np.asarray(packed)]


def satisfies(grid: np.ndarray, clues: list[Clue]) -> bool:
    """Whether a 0/1 grid matches every row and column clue."""
    return grid_clues(grid) == list(clues)


@cache
def line_placements(clue: Clue, n: int) -> np.ndarray:
    """All 0/1 lines of length ``n`` with the given runs, as an int8 array ``[P, n]``.

    Raises ``ValueError`` if the clue cannot fit (so does not return an empty array).
    """
    k = len(clue)
    if k == 0:
        return np.zeros((1, n), dtype=np.int8)
    free = n - (sum(clue) + k - 1)  # cells left over once runs are packed with 1-cell gaps
    if free < 0:
        raise ValueError(f"clue {clue} does not fit in {n} cells")
    rows = []
    # Choose how many of the `free` spare cells precede each run (stars and bars).
    for bars in combinations(range(free + k), k):
        line = np.zeros(n, dtype=np.int8)
        pos = 0
        for i, b in enumerate(bars):
            pos += b - (bars[i - 1] + 1 if i else 0)  # spare cells before run i
            line[pos : pos + clue[i]] = 1
            pos += clue[i] + 1
        rows.append(line)
    return np.stack(rows)


def _lines(state: np.ndarray, n: int) -> list[np.ndarray]:
    """Views on the ``2n`` lines of a state (rows, then columns)."""
    return [state[i] for i in range(n)] + [state[:, j] for j in range(n)]


def _deduce(line: np.ndarray, cands: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Filter a line's placements by its known cells; return (kept placements, forced values)."""
    known = line != UNKNOWN
    if known.any():
        cands = cands[(cands[:, known] == line[known]).all(axis=1)]
    if len(cands) == 0:
        return None
    forced = np.full(line.shape, UNKNOWN, dtype=np.int8)
    forced[cands.min(axis=0) == 1] = 1
    forced[cands.max(axis=0) == 0] = 0
    return cands, forced


def propagate(
    state: np.ndarray, cands: list[np.ndarray], *, parallel: bool = False
) -> tuple[np.ndarray, list[np.ndarray], int] | None:
    """Line-logic fixed point. ``state`` is ``[n, n]`` int8 in ``{-1, 0, 1}`` and is modified.

    Returns ``(state, cands, rounds)`` where ``rounds`` counts the sweeps that fixed at least one
    cell, or ``None`` on a contradiction. ``parallel=True`` makes every line deduce from the same
    snapshot (synchronous rounds); otherwise lines are swept in order and see earlier updates.
    """
    n = state.shape[0]
    rounds = 0
    while True:
        changed = False
        if parallel:
            snapshot = state.copy()
            new_state = state.copy()
            for k, line in enumerate(_lines(snapshot, n)):
                res = _deduce(line, cands[k])
                if res is None:
                    return None
                cands[k], forced = res
                target = _lines(new_state, n)[k]
                upd = forced != UNKNOWN
                if (upd & (target != UNKNOWN) & (target != forced)).any():
                    return None  # a row and a column forced the same cell to different values
                fresh = upd & (target == UNKNOWN)
                if fresh.any():
                    target[fresh] = forced[fresh]
                    changed = True
            state[...] = new_state
        else:
            for k in range(2 * n):
                line = _lines(state, n)[k]
                res = _deduce(line, cands[k])
                if res is None:
                    return None
                cands[k], forced = res
                fresh = (forced != UNKNOWN) & (line == UNKNOWN)
                if fresh.any():
                    line[fresh] = forced[fresh]
                    changed = True
        if not changed:
            return state, cands, rounds
        rounds += 1


@dataclass(frozen=True)
class SolveResult:
    num_solutions: int  # capped at the requested limit
    line_solvable: bool  # line logic alone fixes every cell (implies a unique solution)
    rounds: int  # parallel line-logic rounds to the fixed point (from the empty board)
    solution: np.ndarray | None  # the solution when it is unique, else None


def _count(state: np.ndarray, cands: list[np.ndarray], limit: int, found: list) -> int:
    res = propagate(state, cands)
    if res is None:
        return 0
    state, cands, _ = res
    unknown = np.argwhere(state == UNKNOWN)
    if len(unknown) == 0:
        found.append(state.copy())
        return 1
    i, j = map(int, unknown[0])
    total = 0
    for v in (1, 0):
        branch = state.copy()
        branch[i, j] = v
        total += _count(branch, [c.copy() for c in cands], limit - total, found)
        if total >= limit:
            break
    return total


def solve(clues: list[Clue], n: int, limit: int = 2) -> SolveResult:
    """Count solutions (up to ``limit``) and classify the puzzle."""
    try:
        cands = [line_placements(tuple(c), n) for c in clues]
    except ValueError:
        return SolveResult(0, False, 0, None)
    empty = np.full((n, n), UNKNOWN, dtype=np.int8)

    res = propagate(empty.copy(), [c.copy() for c in cands], parallel=True)
    if res is None:
        return SolveResult(0, False, 0, None)
    state, _, rounds = res
    if (state != UNKNOWN).all():
        return SolveResult(1, True, rounds, state.astype(np.uint8))

    found: list[np.ndarray] = []
    count = _count(empty.copy(), [c.copy() for c in cands], limit, found)
    solution = found[0].astype(np.uint8) if count == 1 else None
    return SolveResult(min(count, limit), False, rounds, solution)
