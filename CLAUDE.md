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
`picross_sheaf`, `picross_mpnn` (param-matched), `picross_mpnn_comm` (comm-matched).
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

### Picross 10x10 (local task): `picross_sheaf` vs `picross_mpnn`

Data: `build_picross` keeps the unique-solution boards of NonoDataset 10x10 (CC BY 4.0): `train`
274,361 (icons + noise/shapes, 64 % filled), `test_easy` 8,847 (icons, solved by line logic alone),
`test_hard` 501 (icons, unique but need search, 32 % filled). Agents: 10 rows + 10 columns on
K(10,10), one edge per cell. Trivial "all empty": `cell_acc` 52.0 / 67.8 %, `solved` 0 %
(easy / hard), so read `solved` / `line_ok`, not `cell_acc`, on `test_hard`.

Runs (seed 42, 5 epochs = 10.7k steps, batch 128, shipped configs, parameter-matched):
- MPNN `outputs/2026-11-13/17-34-30/` (613K params, lr 3e-4, 20/50 rounds): 16 min.
- Sheaf `outputs/2026-11-13/17-50-32/` (597K params, lr 1.7e-3, K 20/50): 63 min, 6.2 GiB.

`solved` (%), EMA weights:

| epoch | Sheaf easy | MPNN easy | Sheaf hard | MPNN hard |
|---|---|---|---|---|
| 0 | 79.8 | 20.4 | 17.9 | 0.0 |
| 1 | 93.0 | 43.1 | 36.7 | 1.2 |
| 2 | 95.3 | 58.1 | 40.1 | 3.6 |
| 3 | 95.9 | 64.7 | 45.0 | 5.6 |
| 4 | **96.4** | 64.4 | **46.4** | 5.2 |

Epoch 4 `line_ok` (fraction of satisfied line clues): Sheaf 99.5 / 85.7, MPNN 92.4 / 60.6.

These are the numbers `train.py` prints, i.e. MPNN evaluated at 50 rounds, beyond its trained 20,
where it degrades: they overstate the gap. Fair comparison below.

MPNN with the Sheaf lr (`training.lr=1.7e-3`): `outputs/2026-11-13/18-58-34/` (16 min); at 50
rounds 87.4 / 31.5 % (easy / hard) at epoch 4.

Communication-matched MPNN (`picross_mpnn_comm`: same d_v 320 and message size 32 as the Sheaf
stalks, hence 1.45M params = 2.4x, lr 1.7e-3): `outputs/2026-11-13/19-42-00/` (22 min); at 50
rounds 97.2 / 66.8 % at epoch 4. The param-matched MPNN has half the Sheaf's per-cell state
(d_v 160 = 16 per cell vs 32).

`scripts/analyze_picross.py` (board by board, all K; outputs in `outputs/analysis_picross/`),
`solved` (%) vs K at eval (all three trained at K = 20, fixed):

| K | 1 | 3 | 5 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| Sheaf easy | 5.3 | 25.2 | 45.7 | 74.4 | 88.5 | **94.6** | 96.2 | **96.4** | 96.0 |
| MPNN comm easy | 0.8 | 5.0 | 6.7 | 49.6 | 89.5 | 95.2 | 96.7 | **97.2** | 97.1 |
| MPNN lr 1.7e-3 easy | 1.0 | 1.8 | 2.5 | 18.3 | 81.0 | 88.3 | **89.1** | 87.4 | 82.4 |
| MPNN lr 3e-4 easy | 0.8 | 4.7 | 11.5 | 49.0 | 80.5 | **87.0** | 81.4 | 64.3 | 51.1 |
| Sheaf hard | 0 | 0 | 1.0 | 14.4 | 30.9 | 44.7 | 44.5 | **46.5** | 45.5 |
| MPNN comm hard | 0 | 0 | 0 | 1.0 | 32.3 | 49.1 | 60.7 | 66.7 | **68.5** |
| MPNN lr 1.7e-3 hard | 0 | 0 | 0 | 0.4 | 20.8 | **37.1** | 37.1 | 31.5 | 20.4 |
| MPNN lr 3e-4 hard | 0 | 0 | 0 | 1.4 | 24.0 | **32.1** | 20.6 | 5.2 | 2.6 |

Takeaways:
- Param-matched (597K vs 613K, same lr), Sheaf wins: at K = 20, 94.6 vs 88.3 (easy), 44.7 vs
  37.1 (hard); the gap grows with board depth (easy boards with 11+ line-logic rounds: 48 vs 29).
  Most of the gap first printed by `train.py` came from the MPNN lr and from evaluating it at 50.
- Communication-matched (same state size, MPNN 2.4x params), the MPNN wins: about equal on easy
  (95.2 vs 94.6 at K = 20, 97.2 vs 96.4 at K = 50) and clearly better on hard (49.1 vs 44.7 at
  K = 20, 66.7 vs 46.5 at K = 50). Same ordering as the paper's Sudoku table (param < comm).
- Stability beyond the trained K is not Sheaf-specific: the comm MPNN is stable too and keeps
  improving on `test_hard` up to K = 100 (68.5 %); only the small (param-matched) MPNNs degrade
  (9 % / 42 % of solved boards lost at K = 100). Sheaf plateaus at ~46 % on hard.
- What stays Sheaf-specific is the *anytime* behaviour: its iterations act like propagation
  rounds (median smallest solving K about 1-2x the board's line-logic rounds: 1->1, 2->2, 3->5,
  4->7, 5->10), so it is already useful at small K (K = 5: 45.7 % easy vs <= 11.5 % for all
  MPNNs); every MPNN needs >= 10-15 rounds even for 2-round boards.
- On `test_hard`, from K ~ 7 on, almost all remaining Sheaf errors lie in the cells where line
  logic stalls; it then resolves them (or not: failures are boards with large stall zones).

Caveats: one seed; 5 epochs; hyperparameters are Sudoku's; train/test distributions differ
(noise boards only in train); no Sheaf variant with more capacity was tried.

Next: seeds 123/456 for Sheaf, MPNN param and MPNN comm; a larger Sheaf (e.g. enc_d_model or
dec_hidden_dims up to ~1.45M params) to separate capacity from architecture; longer training.

#### Learning-rate sweep (2026-11-27): `outputs/lr_sweep/` (`scripts/sweep_lr.py`)

`build_picross` now also holds out `val_easy` (4,000) / `val_hard` (1,000) from the unique train
boards (train 274,361 -> 269,361; test splits unchanged; the runs above used the old train, which
contains the val boards). Sweep: 1 epoch, eval at K = 20 on the val splits only, `ema_decay` 0.995
(0.999 leaves ~12 % init after 2.1k steps), seed 42. `solved` % (val_easy / val_hard):

| model | 3e-4 | 1e-3 | 1.7e-3 | 3e-3 | 5e-3 | 8e-3 |
|---|---|---|---|---|---|---|
| MPNN param | 45.3 / 1.4 | 71.5 / 5.9 | **74.6 / 7.2** | 73.5 / 5.6 | | |
| MPNN comm | 79.5 / 9.5 | **82.2 / 11.7** | 74.5 / 6.2 | 71.0 / 4.1 | | |
| Sheaf | 68.1 / 4.8 | 85.8 / 14.6 | 88.2 / 16.7 | **90.0 / 19.2** | 90.6 / 19.5 | 91.1 / 20.0 |

Chosen lr: MPNN param 1.7e-3 (unchanged), MPNN comm 1e-3 (was 1.7e-3), Sheaf 3e-3 (was 1.7e-3):
Sheaf plateaus from 3e-3 to 8e-3 (within ~1 point, i.e. within noise on 1,000 hard boards), and
one epoch favours high lr, so the lowest lr of the plateau is safer for long constant-lr runs.
After one epoch at its best lr, Sheaf leads on val; over 5 epochs the comm MPNN finished ahead on
test, so the long runs decide.

- Gotcha: Windows sleep suspends WSL training (the run resumes, its wall time counts the sleep)
  or kills it when the session ends; `sweep_lr.py` skips finished runs, so just relaunch it.

#### 10 epochs with the tuned lr (2026-11-27): `outputs/long10/` (`outputs/long10_runs.sh`)

Seed 42, 10 epochs (~21k steps), new train split, eval per epoch at K = 20 on val and test. Wall
time: MPNN param ~30 min, MPNN comm ~45 min, Sheaf ~2h05. `scripts/analyze_picross.py` output in
`outputs/analysis_picross_long10/`. `solved` (%) vs K at eval (all trained at K = 20):

| K | 1 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Sheaf (lr 3e-3) easy | 6.2 | 25.3 | **49.0** | **61.7** | 75.7 | 90.2 | 96.1 | 97.0 | 91.7 | 92.1 |
| MPNN comm (lr 1e-3) easy | 0.1 | 3.4 | 12.4 | 24.8 | 75.4 | 95.8 | **98.6** | 99.3 | 99.5 | 99.5 |
| MPNN param (lr 1.7e-3) easy | 0.3 | 3.2 | 12.4 | 15.2 | 45.8 | 90.5 | 96.1 | 97.4 | 97.5 | 97.2 |
| Sheaf hard | 0 | 0.2 | 0.8 | 4.4 | 14.4 | 30.9 | 47.9 | 46.9 | 31.5 | 34.5 |
| MPNN comm hard | 0 | 0 | 0 | 0 | 13.2 | 52.7 | **70.1** | 82.2 | 87.2 | **89.4** |
| MPNN param hard | 0 | 0 | 0 | 0 | 1.8 | 34.3 | 54.9 | 64.3 | 66.5 | 64.7 |

Per epoch (`test_hard` at K = 20): Sheaf 26.8, 37.5, 41.2, 43.0, 44.6, 44.2, 45.2, 47.2, 45.8,
47.8 (plateau from epoch ~4, train loss still falling to 0.004); MPNN param 1.4 -> 54.8 and MPNN
comm 5.5 -> 70.0, both still rising at epoch 9 (comm slowing).

Takeaways (supersede the 5-epoch ones above):
- With a tuned lr and 10 epochs, both MPNNs beat Sheaf on `test_hard`, the param-matched one
  included (54.9 vs 47.9 at K = 20; 66.5 vs 31.5 at K = 50). On `test_easy` at K = 20, Sheaf ties
  the param-matched MPNN (96.1) and trails the comm-matched one (98.6). The 5-epoch "Sheaf wins
  param-matched" result was an artifact of under-training the MPNN (and of its lr).
- Sheaf learns fastest (26.8 % hard after one epoch vs 1.4 / 5.5 %) but hits a ceiling on hard
  boards (~45-48 %) while its train loss keeps falling: a generalization limit, not lack of
  training. Hard-board depth does not explain it (Sheaf is below the MPNNs in every depth bin).
- Stability beyond the trained K flipped: the 10-epoch MPNNs keep improving up to K = 50-100
  (comm hard 70.1 -> 89.4), while Sheaf at lr 3e-3 degrades beyond K = 30 (hard 47.9 -> 31.5,
  easy 96.1 -> 91.7; 5.7 % of easy boards solved at some K are lost at K = 100). The 5-epoch
  Sheaf at lr 1.7e-3 was stable, so the higher lr may be the cause (untested).
- What remains Sheaf-specific is the anytime behaviour: at K <= 7 it is far ahead (easy K = 5:
  49.0 vs 12.4 %); its median solving K still follows the line-logic rounds (1->1, 2->2, 3->5,
  4->7, 5->10) while the MPNNs need >= 7-10 rounds even for 2-round boards.

Caveats: one seed (val_hard and test_hard have 1,000 / 501 boards, i.e. a few points of noise);
lr chosen on 1-epoch runs; Sheaf hyperparameters other than lr (gamma, rho_init, cg_iters, d_e)
are Sudoku's and untuned; no Sheaf of 1.45M params.

Next: seeds 123/456 on these configs; Sheaf at lr 1.7e-3 for 10 epochs (stability beyond K);
Sheaf with uniform K sampling (`train_iters_dist=uniform`) or K_train 30; a larger Sheaf.
