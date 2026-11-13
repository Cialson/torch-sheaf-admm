# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Unofficial educational PyTorch port of Sheaf-ADMM (Sakana AI, ICML 2026; original JAX code at
https://github.com/SakanaAI/sheaf-admm). The model and ADMM update rules must stay faithful to the
JAX original: parity is checked against frozen JAX outputs in `validation/*.npz`. The import path is
`sheaf_admm`, the distribution name is `torch-sheaf-admm`.

## Commands

Everything goes through `uv` (Python 3.12, see `.python-version`). On Linux the lockfile pins the
PyTorch CUDA 13.0 wheel (`torch==2.14.0` from the `pytorch-cu130` index).

```bash
uv sync                                   # install (add --extra reference only for JAX/Flax/Optax)
uv run python -m ruff check .             # lint (line length 100, rules E,F,I,UP,B,SIM)
uv run python -m pytest -q                # all tests
uv run python -m pytest -q -m "not slow"  # skip end-to-end overfit and compile-parity tests
uv run python -m pytest -q tests/test_admm.py::test_name   # single test
```

Validation scripts (Torch only; the JAX reference fixtures are bundled):

```bash
uv run python validation/check_reference.py --device cuda       # ADMM/solvers vs frozen JAX outputs
uv run python validation/check_lora_reference.py --device cuda  # LoRA maps + gradients vs JAX
uv run python validation/smoke_all_configs.py --device cuda     # 1 train + eval step per config
uv run python validation/check_cli_tasks.py                     # CLIs on synthetic MNIST/Sudoku data
```

Data, training and visualization:

```bash
uv run python -m sheaf_admm.data.build_maze --height 19 --width 19 --train-size 10000 \
  --test-size 1000 --min-path-length 18 --ood-sizes --output-dir datasets/maze_std3_19px_10k
uv run python -m sheaf_admm.data.build_mnist --output-dir datasets/mnist
uv run python -m sheaf_admm.data.build_sudoku --output-dir datasets/sudoku_easy  # ~2.2 GB HF download
uv run python -m sheaf_admm.data.build_picross   # NonoDataset 10x10 (~75 MB, into datasets/picross_10x10/raw), ~6 min

uv run python scripts/train.py +experiment=sudoku_sheaf training.batch_size=32 training.epochs=2
uv run python -m scripts.visualize --checkpoint outputs/<date>/<time>/checkpoint.pt --out-dir viz

# local Game of Life task (see "Game of Life" below)
uv run python -m sheaf_admm.data.build_life        # datasets/life_15px_1step (train, test, test_2x)
uv run python scripts/train.py +experiment=life_sheaf
uv run python scripts/eval_iters.py outputs/<date>/<time>/checkpoint.pt --iters 1 5 10 20 50 100 200

# local Picross (nonogram) 10x10 task (see "Picross" below)
uv run python scripts/show_picross.py --split test_easy --index 41   # data / agents / line logic
uv run python scripts/train.py +experiment=picross_sheaf             # ~15 min/epoch, 6.2 GiB
uv run python scripts/train.py +experiment=picross_mpnn              # ~3 min/epoch
```

`scripts/eval_iters.py` re-evaluates a puzzle-task checkpoint (EMA weights by default,
`--params live` otherwise) at several ADMM iteration / MPNN round counts without retraining.

Experiments: `maze_sheaf`, `maze_mpnn`, `mnist_sheaf`, `mnist_mpnn`, `sudoku_sheaf`,
`sudoku_sheaf_lora`, `sudoku_mpnn` (paper), plus the local `life_sheaf`, `life_mpnn`,
`picross_sheaf`, `picross_mpnn`.
Paper seeds: `training.seed=42|123|456`. W&B is off by default
(`wandb.mode=online` to enable). Outputs (`checkpoint.pt`, `history.json`) go to Hydra's
`outputs/<date>/<time>/`.

## Architecture

The pipeline: **task.prepare → encoder → geometry → unrolled ADMM (x-solver / z-solver) → decoder →
windowed loss**.

- **Config** (`configs/config.yaml` + `configs/experiment/*.yaml`, selected with `+experiment=`):
  `task` picks the task hook, `model_type` picks `sheaf` or `mpnn`, `model:` is turned into a
  `ModelConfig` dataclass (`models/config.py`), `task_cfg:` is forwarded to `make_task`. Solvers,
  restriction-map sharing/init and LoRA mode are all chosen by strings in `model:`.
- **Tasks** (`training/tasks.py`: `MazeTask`, `MNISTTask`, `SudokuTask`): turn a raw numpy batch into
  `fwd` (`patches [N, B, ...]`, `edge_indices [E, 2]`, `model_kwargs`) and `targets`, and own the loss
  and metrics. Maze/MNIST agents are overlapping grid patches (`data/views.py`); Sudoku uses 27
  row/col/box agents on a multigraph.
- **ADMM core** (`admm.py`): states `x, z, y` of shape `[N, B, d_v]`, unrolled for `K` iterations with
  backprop through the whole trajectory (optional truncated BPTT via `grad_window`). Only the last
  `loss_window` x-iterates are returned, then decoded and averaged in the loss. `K_train` and
  `K_eval` differ (e.g. 20 vs 50); Maze can sample `K ~ U[train_iters_min, K_train]`.
- **Solvers** (`solvers/`): a protocol and a name→`(cls, Params)` registry for each family.
  x-solvers do the local prox step (`diagonal_prox` is what the shipped sheaf configs use);
  z-solvers do the sheaf consensus step (`unrolled_cg` default, `gd` ablation). `z_mode` is
  `project` (hard consensus) or `prox`.
- **Geometry** (`geometry/`): `SheafGeometry` protocol, restriction maps `[E, 2, d_e, d_v]`,
  matrix-free sheaf Laplacian (`laplacian_apply`, used by CG) and energy (used by GD).
  `FixedGeometry` = learned but input-independent maps (`rm_mode: fixed`); `LoRAGeometry` =
  input-modulated `F = R + (alpha/r) A Bᵀ` (`rm_mode: context`).
- **Models** (`models/`): encoder and decoder are shared across agents (the `N` axis is flattened into
  the batch). `SheafADMMModel` wires everything; `MPNNModel` is the recurrent message-passing
  baseline reusing the same encoder/decoder.
- **Training** (`training/loop.py`, `optim.py`): AdamW + linear warmup + grad clipping, EMA weights
  used at eval (`_ema_model` swaps them in temporarily). `scripts/train.py` is the single Hydra
  entrypoint for all tasks and both model families.

## Gotchas

- Importing `sheaf_admm` calls `set_high_precision()` (TF32 off, `"highest"` matmul precision). The
  unrolled-CG equivalence tests and JAX parity depend on it; keep it.
- On CUDA, the ADMM step (`admm.py`) and the GGNN round (`models/mpnn.py`) are wrapped in
  `torch.compile(mode="reduce-overhead")`, i.e. CUDA graphs. `TORCH_COMPILE_DISABLE=1` falls back
  to eager. Triton compiles a C helper at first use, so the Python headers must be installed
  (`python3.12-dev`, or use `uv sync --managed-python`); otherwise the compile-parity and
  end-to-end tests fail with `Python.h: No such file or directory`.
- Memory: on 8 GB GPUs, `sudoku_sheaf` with the default `batch_size: 128` fills VRAM (on Windows/WSL
  the driver may then silently spill to system RAM, so training stalls at 0 % GPU use instead of
  raising OOM). `training.batch_size=32` uses about 4 GB.
- `train.py` prints only at the end of each validated epoch, so long silent periods are normal.
- Runs from separate initializations do not match the JAX version bit for bit (different RNGs);
  parity is only guaranteed for the fixture-covered cases.
- `scripts/convert_checkpoint.py` (legacy JAX Maze Sheaf `.pkl` → `.pt`) needs `--extra reference`
  and unpickles; only use it on trusted files. The `scripts/visualize.py` docstring still mentions
  `checkpoint.pkl`, but the script loads the `.pt` checkpoints written by `train.py`.
- `train.py` writes `checkpoint.pt` / `history.json` only once, after the last epoch (no
  intermediate checkpoints). Use `training.epochs=N` for a shorter run you can visualize.
- `scripts/visualize.py` only supports Sheaf checkpoints (MPNN has no ADMM trajectory); compare
  MPNN runs through `history.json` metrics.
- `task:` in a config is a key of the `make_task` registry (`maze | mnist | sudoku | picross`), not a free
  experiment name; a new grid task can reuse `task: maze` (see Game of Life).
- Eval uses EMA weights. With the default `ema_decay: 0.999`, short runs (< ~3000 steps) report
  metrics of weights still dominated by the random init (780 steps: 0.999^780 ≈ 46 % init). Use a
  smaller decay (e.g. 0.995) for short runs, or compare with `eval_iters.py --params live`.
- Sheaf and MPNN configs need different `model:` blocks: the parameter-matched MPNN uses
  `d_v: 84, d_e: 42, mpnn_message_dim: 42` (~182K params, like the `d_v: 10` Sheaf). Copying the
  Sheaf block into an MPNN config silently gives a ~49K-param MPNN (Sheaf-only keys are ignored).
  Check the `[init] ... params=` line of both runs.
- `train.py` only samples `K ~ U[train_iters_min, K_train]` for Sheaf; MPNN always trains with a
  fixed `mpnn_train_rounds`.
- The shipped `sudoku_mpnn` is *not* parameter-matched (~3.0M params vs ~543K for `sudoku_sheaf`);
  `picross_mpnn` is (`d_v: 160`, ~613K vs ~597K for `picross_sheaf`).
- Slot-based models (`encoder_arch: sudoku`, Sudoku and Picross) are sized by `model.num_slots`
  (cells per agent), `num_cells` (board cells) and `slot_input_dim` (features per slot); defaults
  are Sudoku's (9 / 81 / 10). A mismatch with the task raises at the first forward.

## Local results (RTX 4060 Laptop 8 GB, WSL2)

### Maze Sheaf, 10 epochs: `outputs/2026-11-06/17-41-12/`

Command: `uv run python scripts/train.py +experiment=maze_sheaf training.epochs=10` (seed 42,
everything else from `maze_sheaf.yaml`). Wall time about 37 min (17:41 → 18:18), so about
3.7 min/epoch including validation every 5 epochs. Figures in `viz/` (from `scripts.visualize`).

`solved` (whole maze exactly right) / `cell_acc`, EMA weights, `K_eval=100`:

| split (size) | epoch 0 | epoch 5 | epoch 9 | paper (50 ep.) |
|---|---|---|---|---|
| test (19×19) | 0 / 74.3 | 97.81 / 99.97 | 98.12 / 99.98 | 99.9 |
| test_ood_2x (37×37) | 0 / 75.3 | 73.30 / 99.37 | 91.93 / 99.84 | 98.1 |
| test_ood_2xW (19×37) | 0 / 74.9 | 94.59 / 99.92 | 95.71 / 99.95 | — |
| test_ood_4x (73×73) | 0 / 75.9 | 6.11 / 93.42 | 3.14 / 94.45 | 4.5 |
| test_ood_4xW (19×73) | 0 / 75.5 | 76.63 / 99.58 | 91.46 / 99.83 | — |

Train loss reaches 3e-4 by epoch 5. Takeaways:
- 10 epochs already gets close to the paper's 50-epoch numbers; `2x` is still clearly improving
  (73 → 92 %), so a full 50-epoch run should close most of the gap.
- `4x` failing is expected (the paper gets 4.5 %): cell accuracy stays at 94 %, but on a 73×73
  board a handful of wrong cells fails the maze, and the path is much longer than what 100
  iterations propagate. The wide-only `4xW` generalizes well (91 %).

Visualizations (first test batch, sample 0):
- `prediction_evolution`: k=1 gives isolated local fragments near start/goal/corridors; the path
  grows from both endpoints, spurious fragments (k=5) are pruned by k=10, and the path is complete
  and clean at k=30. Information spreads gradually through consensus.
- `coordination_dynamics` (learned rho = 0.43): the mean primal residual falls from about 2 to
  0.03 and the mean dual from about 1.5 to 0.01 over 30 iterations. A few agents (likely on the
  path or at junctions) converge more slowly; the max primal residual stays around 0.1. The
  sheaf-consistency RMS stays flat around 2; this is expected with `z_mode: prox` (soft
  consensus, gamma=5), since z is not forced onto global sections. Worth checking whether it
  drops after 50 epochs.
- `xz_trajectories`: for most agents x snaps onto z within 1–2 iterations, then both move together
  to the fixed point. A few agents (4, 17, 34, 38, 61) start far away and are pulled back.

### Maze MPNN, 10 epochs: `outputs/2026-11-06/18-32-44/`

Command: `uv run python scripts/train.py +experiment=maze_mpnn training.epochs=10` (seed 42,
parameter-matched baseline, ~182K params, max aggregation, 40 train / 100 eval rounds). Wall time
about 16.5 min (18:32 → 18:49), about 2.3× faster than Sheaf. The warning
`CUDAGraph ... observed 9 distinct sizes` is harmless: each OOD split has a different number of
agents, so a CUDA graph is recorded per graph size.

`solved` (%) at epoch 9, Sheaf vs MPNN (same seed, same epochs):

| split | Sheaf | MPNN | MPNN epoch 5 |
|---|---|---|---|
| test (19×19) | 98.12 | **100.00** | 100.00 |
| test_ood_2x (37×37) | **91.93** | 56.72 | 60.31 |
| test_ood_2xW (19×37) | 95.71 | **99.90** | 99.90 |
| test_ood_4x (73×73) | **3.14** | 0.29 | 1.71 |
| test_ood_4xW (19×73) | **91.46** | 72.58 | 75.38 |

Takeaways:
- MPNN wins at training size and on `2xW` (same height as training), but loses clearly when the
  board grows in both directions (`2x`: 57 vs 92 %) or gets much wider (`4xW`: 73 vs 91 %).
- MPNN OOD scores *decrease* from epoch 5 to 9 while in-distribution is saturated at 100 %
  (overfitting to the training size); Sheaf OOD scores keep increasing. This matches the paper's
  claim that the ADMM/sheaf structure helps size generalization.
- Caveats: one seed, 10 of 50 epochs; the printed `loss` is the last training batch only (noisy).
  Confirming needs seeds 123/456 and/or full 50-epoch runs.

### Game of Life (local task): `life_sheaf` vs `life_mpnn`

Task: predict the next generation of a random 15×15 grid (30 % alive, cells outside are dead).
`src/sheaf_admm/data/build_life.py` writes the maze puzzle format (10k train, 1k `test`, 1k
`test_2x` at 31×31, one example per puzzle/group). It reuses `MazeTask` unchanged through the token
encoding dead=2, alive=5: `MazeTask` pads inputs with wall=1 and zero-pads label patches, and its
`solved` metric compares `PATH_TOKEN=5` cells, which for a binary grid means the whole grid is
exact. Use odd sizes so stride-2 agents tile the grid (15×15 → 49 agents / 156 edges; 31×31 → 225
/ 812). Trivial "all dead" baseline: `cell_acc` 68.9 % (test), 67.1 % (test_2x), `solved` 0 %.

Runs (seed 42, 40 epochs = 3120 steps, batch 128, lr 3e-4, `ema_decay` 0.995, ~182K params each):
- Sheaf `outputs/2026-11-07/13-42-36/` (5.5 min): `maze_sheaf` model block, `K ~ U[5, 20]`,
  `K_eval` 20.
- MPNN `outputs/2026-11-07/14-22-46/` (3 min): `maze_mpnn` model block, 20 fixed rounds.
  (`13-48-37` is an invalid MPNN run with the Sheaf model block, ~49K params.)

`solved` (%) from `eval_iters.py`, EMA weights; trained at 20 iterations/rounds:

| K at eval | 1 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| Sheaf test (15×15) | 0 | **95.8** | 95.3 | 94.0 | 49.6 | 10.3 | 3.7 |
| Sheaf test_2x (31×31) | 0 | **94.2** | 94.3 | 90.3 | 2.0 | 0 | 0 |
| MPNN test (15×15) | 0 | 0 | 0 | **99.9** | 0 | 0 | 0 |
| MPNN test_2x (31×31) | 0 | 0 | 0 | **99.7** | 0 | 0 | 0 |

`cell_acc` at K=200: Sheaf 97.2 / 94.8, MPNN 79.0 / 78.1 (test / test_2x).

Takeaways:
- Both learn the rule and generalize to 31×31: per-cell accuracy is even higher at 31×31, and
  the lower `solved` comes only from having 4× more cells (0.9997^225 ≈ 93.5 %).
- At training conditions MPNN is better and faster (74 % vs 2 % `solved` at epoch 10).
- Sheaf is robust over the whole trained K range (5–20; already best at K=5), MPNN only works at
  exactly 20 rounds: its computation is spread over the trained horizon.
- Neither is stable beyond the trained K; Sheaf degrades more gracefully (cell level).
- K=1 gives about 75 % for Sheaf: the decoded `x` of the first iteration is computed before any
  z-exchange, so only patch-center cells are predictable without neighbors.
- Sheaf's remaining errors are concentrated on the grid border (about 10× the interior rate).

Caveats: one seed; K training differs (uniform for Sheaf, fixed for MPNN), so part of Sheaf's
robustness may come from K sampling; Sheaf hyperparameters are Maze's and untuned (`d_e: 5`), and
it was still improving at epoch 39; a one-generation rule is purely local and does not test the
global coordination the paper targets.

Next: (A) Sheaf with `training.train_iters_dist=fixed` (same K training as MPNN); (B) Sheaf with
`training.K_train=40 training.train_iters_min=15 training.K_eval=40` for stability beyond 20;
seeds 123/456 for both; a less local task (`build_life --steps 3`).
