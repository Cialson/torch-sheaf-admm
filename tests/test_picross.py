"""Picross: clues, packing, line solver, builder, line-agent views / graph, task, and models."""

from __future__ import annotations

import json
from math import comb

import numpy as np
import pytest
import torch

from sheaf_admm.data import picross as P
from sheaf_admm.data import views as V
from sheaf_admm.data.build_picross import PicrossConfig, build_from_arrays
from sheaf_admm.data.loaders import PuzzleDataset
from sheaf_admm.geometry import FixedGeometry, build_sudoku_restriction_maps
from sheaf_admm.models import ModelConfig, model_config_from_dict
from sheaf_admm.training import build_model
from sheaf_admm.training.tasks import PicrossTask, make_task, picross_line_clues


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


# ---------------------------------------------------------------------------
# Views and graph (sheaf_admm.data.views)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [4, 10])
def test_picross_graph_is_complete_bipartite_with_one_edge_per_cell(n):
    edges, map_u, map_v = V.build_picross_graph(n)
    assert edges.shape == (n * n, 2) and map_u.shape == map_v.shape == (n * n,)
    assert V.picross_num_nodes(n) == 2 * n
    assert bool((edges[:, 0] < n).all() and (edges[:, 1] >= n).all())  # row -> column
    pairs = {(int(u), int(v)) for u, v in edges}
    assert pairs == {(i, n + j) for i in range(n) for j in range(n)}
    assert int(map_u.max()) == int(map_v.max()) == n - 1


@pytest.mark.parametrize("n", [4, 10])
def test_picross_edge_slots_point_at_the_shared_cell(n):
    """Slot ``map_u`` of the row and slot ``map_v`` of the column are the same cell, and
    edge ``e`` is cell ``e``; every cell is covered by exactly two agents."""
    edges, map_u, map_v = V.build_picross_graph(n)
    cell_ids = V.build_picross_cell_indices(n)
    assert cell_ids.shape == (2 * n, n)
    via_u = cell_ids[edges[:, 0], map_u]
    via_v = cell_ids[edges[:, 1], map_v]
    torch.testing.assert_close(via_u, via_v)
    torch.testing.assert_close(via_u, torch.arange(n * n))
    np.testing.assert_array_equal(np.bincount(cell_ids.reshape(-1), minlength=n * n), 2)


def test_picross_slice_reassemble_roundtrip():
    n, B, C = 6, 3, 4
    grid = torch.randn(B, n, n, C)
    views = V.picross_slice_batch(grid)
    assert views.shape == (B, 2 * n, n, C)
    # Slicing is a gather of the flat grid by the cell ids.
    cell_ids = V.build_picross_cell_indices(n)
    torch.testing.assert_close(views, grid.reshape(B, n * n, C)[:, cell_ids])
    torch.testing.assert_close(V.reassemble_picross_logits(views), grid)


def test_picross_reassemble_averages_row_and_column_views():
    n = 3
    views = torch.zeros(1, 2 * n, n, 1)
    views[0, 1, 2, 0] = 4.0  # row 1, slot 2 -> cell (1, 2)
    views[0, n + 2, 1, 0] = 2.0  # column 2, slot 1 -> same cell
    recon = V.reassemble_picross_logits(views)
    assert recon[0, 1, 2, 0] == 3.0 and recon.sum() == 3.0


def test_picross_patches_carry_each_agent_clue():
    n = 10
    rng = np.random.default_rng(4)
    grids = (rng.random((2, n, n)) < 0.6).astype(np.uint8)
    packed = np.stack([P.pack_clues(P.grid_clues(g), n).reshape(-1) for g in grids])  # disk format
    L = P.clue_slots(n)

    onehot = V.picross_patches(torch.as_tensor(packed), n)
    assert onehot.shape == (2 * n, 2, n, L * (n + 1))
    torch.testing.assert_close(onehot, onehot[:, :, :1].expand_as(onehot))  # same on all slots
    decoded = onehot[:, :, 0].reshape(2 * n, 2, L, n + 1).argmax(-1)  # [2n, B, L]
    for b, g in enumerate(grids):
        clues = P.grid_clues(g)
        for agent in range(2 * n):  # rows then columns, as in the graph
            got = tuple(int(v) for v in decoded[agent, b] if v)
            assert got == clues[agent]

    scalar = V.picross_patches(torch.as_tensor(packed), n, encoding="scalar")
    assert scalar.shape == (2 * n, 2, n, L)
    torch.testing.assert_close(scalar[:, :, 0] * n, decoded.float())
    with pytest.raises(ValueError):
        V.picross_patches(torch.as_tensor(packed), n, encoding="bogus")


def test_picross_consistent_board_is_a_global_section():
    """With slot-selector restriction maps (the Sudoku ``soft_slice`` init without noise), the
    per-agent views of one board have zero sheaf energy; corrupting one agent only breaks the
    edges incident to it. This is the wiring the Sheaf model relies on."""
    n, B, d_e = 5, 2, 3
    d_v = n * d_e
    selectors = torch.zeros(n, d_e, d_v)
    for k in range(n):
        selectors[k, :, k * d_e : (k + 1) * d_e] = torch.eye(d_e)
    edges, map_u, map_v = V.build_picross_graph(n)
    geom = FixedGeometry(edges, build_sudoku_restriction_maps(selectors, map_u, map_v))

    cells = torch.randn(B, n, n, d_e)
    z = V.picross_slice_batch(cells).permute(1, 0, 2, 3).reshape(2 * n, B, d_v)
    assert geom.energy(z) < 1e-10
    assert geom.laplacian_apply(z).abs().max() < 1e-6

    z_bad = z.clone()
    z_bad[n + 2] += 1.0  # corrupt column agent 2
    res = geom.edge_residuals(z_bad).abs().amax(dim=(1, 2))  # [E]
    touched = edges[:, 1] == n + 2
    assert bool((res[touched] > 0.5).all()) and bool((res[~touched] < 1e-6).all())


def test_propagate_round_by_round_reaches_the_same_fixed_point():
    rng = np.random.default_rng(5)
    n = 10
    grid = (rng.random((n, n)) < 0.6).astype(np.uint8)
    cands = [P.line_placements(c, n) for c in P.grid_clues(grid)]
    empty = np.full((n, n), P.UNKNOWN, dtype=np.int8)
    full = P.propagate(empty.copy(), [c.copy() for c in cands], parallel=True)

    state, cs, steps = empty.copy(), [c.copy() for c in cands], 0
    while True:
        state, cs, r = P.propagate(state, cs, parallel=True, max_rounds=1)
        if r == 0:
            break
        assert r == 1
        steps += 1
    np.testing.assert_array_equal(state, full[0])
    assert steps == full[2]


# ---------------------------------------------------------------------------
# Task: prepare / loss / metrics (sheaf_admm.training.tasks.PicrossTask)
# ---------------------------------------------------------------------------


def _batch(n=10, B=4, seed=6):
    rng = np.random.default_rng(seed)
    grids = (rng.random((B, n, n)) < 0.6).astype(np.uint8)
    inputs = np.stack([P.pack_clues(P.grid_clues(g), n).reshape(-1) for g in grids])
    return {"inputs": inputs, "labels": grids.reshape(B, -1), "height": n, "width": n}, grids


def _perfect_logits(grids, scale=10.0):
    """Per-agent logits ``[2n, B, n, 2]`` that put the right class on every slot."""
    views = V.picross_slice_batch(torch.as_tensor(grids, dtype=torch.long)[..., None])[..., 0]
    return (torch.nn.functional.one_hot(views, 2).float() * 2 - 1).permute(1, 0, 2, 3) * scale


@pytest.mark.parametrize("n", [1, 2, 5, 10])
def test_picross_line_clues_matches_numpy_reference(n):
    rng = np.random.default_rng(n)
    lines = (rng.random((200, n)) < 0.5).astype(np.uint8)
    lines[0], lines[1] = 0, 1  # empty and full lines
    got = picross_line_clues(torch.as_tensor(lines)).numpy()
    want = np.stack([P.pack_clues([P.line_clue(x)] * (2 * n), n)[0] for x in lines])
    np.testing.assert_array_equal(got, want)


def test_picross_task_prepare_shapes_and_targets():
    batch, grids = _batch()
    n, B = 10, 4
    task = make_task("picross")
    fwd, targets, aux = task.prepare(batch)
    assert fwd["patches"].shape == (2 * n, B, n, 5 * 11)
    assert fwd["edge_indices"].shape == (n * n, 2)
    assert set(fwd["model_kwargs"]) == {"map_u", "map_v", "cell_ids"}
    assert targets["slice_labels"].shape == (2 * n, B, n)
    g = torch.as_tensor(grids).long()
    torch.testing.assert_close(targets["slice_labels"][3, 1], g[1, 3])  # row agent 3
    torch.testing.assert_close(targets["slice_labels"][n + 6, 1], g[1, :, 6])  # column agent 6
    assert targets["clues"].shape == (B, 2 * n, 5)
    assert make_task("picross", clue_encoding="scalar").prepare(batch)[0]["patches"].shape[-1] == 5


def test_picross_task_loss():
    batch, grids = _batch()
    task = PicrossTask()
    _, targets, _ = task.prepare(batch)
    window = _perfect_logits(grids).unsqueeze(0).expand(2, -1, -1, -1, -1)  # [W=2, 2n, B, n, 2]
    assert task.loss(window, targets) < 1e-6
    uniform = torch.zeros_like(window)
    torch.testing.assert_close(task.loss(uniform, targets), torch.tensor(np.log(2.0)).float())


def test_picross_task_metrics():
    batch, grids = _batch()
    n, B = 10, 4
    task = PicrossTask()
    _, targets, _ = task.prepare(batch)

    logits = _perfect_logits(grids)
    m = task.evaluate(logits, targets, {})
    assert m == {"cell_acc": 1.0, "solved": 1.0, "line_ok": 1.0, "clue_ok": 1.0, "view_agree": 1.0}

    # Flip cell (2, 7) of board 0 in both views: one wrong cell breaks its row and its column.
    both = logits.clone()
    both[2, 0, 7] *= -1
    both[n + 7, 0, 2] *= -1
    m = task.evaluate(both, targets, {})
    assert m["cell_acc"] == pytest.approx(1 - 1 / (B * n * n))
    assert m["solved"] == m["clue_ok"] == pytest.approx(1 - 1 / B)
    assert m["line_ok"] == pytest.approx(1 - 2 / (B * 2 * n))
    assert m["view_agree"] == 1.0

    # Only the row view disagrees, but more confidently: the averaged cell is wrong and the two
    # views no longer agree on it.
    one = logits.clone()
    one[2, 0, 7] *= -3
    m = task.evaluate(one, targets, {})
    assert m["cell_acc"] == pytest.approx(1 - 1 / (B * n * n))
    assert m["view_agree"] == pytest.approx(1 - 1 / (B * n * n))


def test_picross_task_on_built_dataset(tmp_path):
    """Batches straight from the puzzle loader go through prepare / evaluate."""
    n = 8
    rng = np.random.default_rng(7)
    y = (rng.random((120, n * n)) < 0.55).astype(np.uint8)
    x = np.stack([P.pack_clues(P.grid_clues(g.reshape(n, n)), n).reshape(-1) for g in y])
    build_from_arrays(PicrossConfig(size=n, output_dir=tmp_path, workers=1), x, y, x[:0], y[:0])
    _, batch = next(PuzzleDataset(tmp_path, "train").iter_test_batches(16))
    task = PicrossTask()
    fwd, targets, _ = task.prepare(batch)
    grids = targets["labels_grid"].numpy()
    m = task.evaluate(_perfect_logits(grids), targets, {})
    assert m["solved"] == 1.0 and m["clue_ok"] == 1.0


# ---------------------------------------------------------------------------
# Models with num_slots = n (phase 3)
# ---------------------------------------------------------------------------

_SMALL = dict(
    num_classes=2, d_v=40, d_e=4, encoder_arch="sudoku", enc_d_model=16, enc_num_blocks=1,
    decoder_arch="sudoku", dec_hidden_dims=(16,), num_slots=10, num_cells=100,
    slot_input_dim=55, num_directions=10,
)
_SHEAF = dict(
    objective_mode="non_negative", x_solver="diagonal_prox", z_solver="unrolled_cg",
    z_mode="prox", gamma=2.0, cg_iters=3, rm_sharing="sudoku", rm_init="soft_slice",
    comm_norm_type="layernorm",
)
_MODELS = {
    "sheaf_fixed": ("sheaf", {**_SMALL, **_SHEAF, "rm_mode": "fixed"}),
    "sheaf_lora": ("sheaf", {**_SMALL, **_SHEAF, "rm_mode": "context", "lora_rank": 2,
                             "lora_init_style": "standard"}),
    "mpnn": ("mpnn", {**_SMALL, "mpnn_edge_type_mode": "slot", "mpnn_message_dim": 4}),
}


def _run(model, model_type, fwd, k=3, window=2):
    if model_type == "sheaf":
        return model(fwd["patches"], fwd["edge_indices"], num_iters=k, loss_window=window,
                     **fwd["model_kwargs"], training=True, compile_steps=False)[0]
    return model(fwd["patches"], fwd["edge_indices"], num_rounds=k, **fwd["model_kwargs"],
                 training=True, compile_steps=False)[0].unsqueeze(0)


@pytest.mark.parametrize("name", sorted(_MODELS))
def test_picross_models_forward_backward(name):
    model_type, kw = _MODELS[name]
    model = build_model(model_config_from_dict(kw), model_type, seed=0)
    batch, _ = _batch(B=3)
    task = PicrossTask()
    fwd, targets, _ = task.prepare(batch)
    logits = _run(model, model_type, fwd)
    expected_w = 2 if model_type == "sheaf" else 1
    assert logits.shape == (expected_w, 20, 3, 10, 2)
    loss = task.loss(logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # The MPNN reuses the Sheaf encoder, whose ADMM-objective `beta_raw` it never reads (same
    # for the Sudoku MPNN); everything else must receive a finite gradient.
    assert no_grad == (["encoder.objective_heads.beta_raw"] if model_type == "mpnn" else [])
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    metrics = task.evaluate(logits[-1].detach(), targets, {})
    assert 0.0 <= metrics["cell_acc"] <= 1.0


def test_picross_slot_maps_select_their_cell_block():
    """soft_slice with 10 slots: map k is (nearly) the selector of stalk block k."""
    model = build_model(model_config_from_dict(_MODELS["sheaf_fixed"][1]), "sheaf", seed=0)
    R = model.rm()  # [10, d_e, d_v]
    assert R.shape == (10, 4, 40)
    for k in range(10):
        block = R[k, :, 4 * k : 4 * (k + 1)]
        torch.testing.assert_close(block, torch.eye(4), atol=0.05, rtol=0)
        assert R[k].abs().sum() - block.abs().sum() < 0.05 * 36 * 4


def test_picross_mpnn_has_one_edge_type_per_slot():
    model = build_model(model_config_from_dict(_MODELS["mpnn"][1]), "mpnn", seed=0)
    assert model.ggnn.cell.dir_kernel.shape[0] == 10  # one message kernel per slot


@pytest.mark.parametrize(
    "override, match",
    [
        ({"num_slots": 9, "d_v": 36, "num_directions": 9}, "num_slots"),  # Sudoku layout
        ({"slot_input_dim": 5}, "slot_input_dim"),  # scalar clue width expected
        ({"num_cells": 81}, "num_cells"),
    ],
)
def test_picross_model_rejects_mismatched_slot_layout(override, match):
    model = build_model(
        model_config_from_dict({**_MODELS["sheaf_fixed"][1], **override}), "sheaf", seed=0
    )
    fwd, _, _ = PicrossTask().prepare(_batch(B=2)[0])
    with pytest.raises(ValueError, match=match):
        _run(model, "sheaf", fwd)


def test_slot_config_validation():
    with pytest.raises(ValueError, match="multiple of num_slots"):
        ModelConfig(**{**_SMALL, "d_v": 45})
    with pytest.raises(ValueError, match="num_directions == num_slots"):
        ModelConfig(**{**_MODELS["sheaf_lora"][1], "num_directions": 9})
    ModelConfig(d_v=10)  # non slot-based configs are not constrained
