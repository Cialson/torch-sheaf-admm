"""Task-specific tensor preparation, losses, and evaluation metrics."""

from __future__ import annotations

import functools
import math

import numpy as np
import torch
import torch.nn.functional as F

from sheaf_admm.data import views as V

PATH_TOKEN = 5


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.long().reshape(-1), reduction="none"
    ).reshape(labels.shape)


def _grid_graph(image_hw, stride, patch_size, connectivity):
    centers = V.grid_agent_centers(image_hw, stride=stride, patch_size=patch_size)
    edges = torch.as_tensor(
        V.build_grid_edge_indices(centers, stride, connectivity), dtype=torch.long
    )
    positions = torch.as_tensor(V.node_positions(centers), dtype=torch.float32)
    return centers, edges, positions


class MazeTask:
    name = "maze"

    def __init__(self, *, patch_size=3, stride=2, connectivity=8, num_classes=6):
        self.patch_size, self.stride, self.connectivity = patch_size, stride, connectivity
        self.num_classes = num_classes

    def prepare(self, batch):
        inputs = torch.as_tensor(batch["inputs"])
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        B = inputs.shape[0]
        H, W = V.infer_image_hw(inputs.shape[1], batch.get("height"), batch.get("width"))
        inp_img, lab_img = inputs.reshape(B, H, W), labels.reshape(B, H, W)
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.prepare_maze_patches(inp_img, centers, self.patch_size, self.num_classes)
        label_patches, masks = V.get_agent_label_patches_jax(lab_img, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        targets = {"label_patches": label_patches.long(), "masks": masks, "labels_img": lab_img}
        return fwd, targets, {"centers": centers, "image_hw": (H, W)}

    def loss(self, logits_window, targets):
        lp, masks = targets["label_patches"], targets["masks"]
        ce = _ce(logits_window, lp.unsqueeze(0).expand(logits_window.shape[0], *lp.shape))
        return (
            (ce * masks.unsqueeze(0)).sum(dim=tuple(range(1, ce.ndim))) / masks.sum().clamp_min(1)
        ).mean()

    def evaluate(self, logits_final, targets, aux):
        recon = V.reassemble_logits(
            logits_final, aux["centers"], aux["image_hw"], self.num_classes, mode="mean"
        )
        pred = recon.argmax(dim=-1)
        labels = targets["labels_img"]
        solved = ((pred == PATH_TOKEN) == (labels == PATH_TOKEN)).all(dim=(1, 2))
        return {
            "solved": solved.float().mean().item(),
            "cell_acc": (pred == labels).float().mean().item(),
        }


class MNISTTask:
    name = "mnist"

    def __init__(self, *, patch_size=3, stride=3, connectivity=8, num_classes=10):
        self.patch_size, self.stride, self.connectivity, self.num_classes = (
            patch_size,
            stride,
            connectivity,
            num_classes,
        )

    def prepare(self, batch):
        images = torch.as_tensor(batch["images"], dtype=torch.float32)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        H, W = images.shape[1:3]
        centers, edges, npos = _grid_graph((H, W), self.stride, self.patch_size, self.connectivity)
        patches = V.patchify_batch_jax(images, centers, self.patch_size)
        fwd = {"patches": patches, "edge_indices": edges, "model_kwargs": {"node_positions": npos}}
        return fwd, {"labels": labels}, {}

    def loss(self, logits_window, targets):
        labels = targets["labels"]
        expanded = labels.view(1, 1, -1).expand(logits_window.shape[:-1])
        return _ce(logits_window, expanded).mean()

    def loss_graph(self, logits, targets):
        return _ce(logits, targets["labels"]).mean()

    def evaluate(self, logits_final, targets, aux):
        agg = logits_final.softmax(dim=-1).mean(dim=0)
        return {"acc": (agg.argmax(dim=-1) == targets["labels"]).float().mean().item()}

    def evaluate_graph(self, logits, targets, aux):
        return {"acc": (logits.argmax(dim=-1) == targets["labels"]).float().mean().item()}


class SudokuTask:
    name = "sudoku"
    num_classes = 10

    def prepare(self, batch):
        inputs = torch.as_tensor(batch["inputs"], dtype=torch.long)
        labels = torch.as_tensor(batch["labels"], dtype=torch.long)
        B = inputs.shape[0]
        inp, lab = inputs.reshape(B, 9, 9), labels.reshape(B, 9, 9)
        edges, map_u, map_v = V.build_sudoku_multigraph(9)
        cell_ids = V.build_sudoku_cell_indices(9)
        patches = V.sudoku_slice_batch_jax(F.one_hot(inp, self.num_classes).float()).permute(
            1, 0, 2, 3
        )
        slice_labels = V.sudoku_slice_batch_jax(lab[..., None])[..., 0].permute(1, 0, 2).long()
        fwd = {
            "patches": patches,
            "edge_indices": torch.as_tensor(edges, dtype=torch.long),
            "model_kwargs": {
                "map_u": torch.as_tensor(map_u, dtype=torch.long),
                "map_v": torch.as_tensor(map_v, dtype=torch.long),
                "cell_ids": torch.as_tensor(cell_ids, dtype=torch.long),
            },
        }
        targets = {"slice_labels": slice_labels, "labels_grid": lab, "inputs_grid": inp}
        return fwd, targets, {}

    def loss(self, logits_window, targets):
        labels = targets["slice_labels"]
        return _ce(logits_window, labels.unsqueeze(0).expand(logits_window.shape[:-1])).mean()

    def evaluate(self, logits_final, targets, aux):
        recon = V.reassemble_sudoku_logits(logits_final.permute(1, 0, 2, 3))
        pred = recon.argmax(dim=-1)
        labels = targets["labels_grid"]
        empty = targets["inputs_grid"] == 0
        correct = pred == labels
        return {
            "cell_acc": correct.float().mean().item(),
            "solved": correct.all(dim=(1, 2)).float().mean().item(),
            "completion": (correct & empty).sum().float().div(empty.sum().clamp_min(1)).item(),
        }


def picross_line_clues(lines: torch.Tensor) -> torch.Tensor:
    """Packed clues of 0/1 lines: ``[..., n]`` -> ``[..., ceil(n/2)]`` long, run lengths
    right-aligned and left-padded with 0 (the dataset input layout, see ``data.picross``)."""
    x = lines.long()
    n = x.shape[-1]
    slots = (n + 1) // 2  # a line of n cells has at most ceil(n/2) runs
    prev = F.pad(x[..., :-1], (1, 0))
    starts = x * (1 - prev)  # 1 where a run begins
    num_runs = starts.sum(-1, keepdim=True)  # [..., 1]
    run_idx = (starts.cumsum(-1) - 1).clamp_min(0)  # run each filled cell belongs to
    lengths = torch.zeros(*x.shape[:-1], slots, dtype=torch.long, device=x.device)
    lengths = lengths.scatter_add(-1, run_idx, x)  # [..., slots], left-aligned
    shift = slots - num_runs  # right-align: run r goes to slot shift + r
    src = torch.arange(slots, device=x.device) - shift  # [..., slots]
    packed = lengths.gather(-1, src.clamp_min(0))
    return torch.where(src >= 0, packed, torch.zeros_like(packed))


class PicrossTask:
    """Nonogram: ``2n`` line agents (rows, then columns) on ``K(n, n)``, see ``data.views``.

    Batches come from ``build_picross``: ``inputs`` ``[B, n*n]`` packed clues, ``labels``
    ``[B, n*n]`` the 0/1 solution. The model predicts 2 classes per agent slot; the loss is the
    cross-entropy of every agent's ``n`` slots (each cell is supervised in its row and its column
    view), as for Sudoku.
    """

    name = "picross"
    num_classes = 2

    def __init__(self, *, clue_encoding: str = "onehot"):
        self.clue_encoding = clue_encoding

    @staticmethod
    @functools.cache
    def _graph(n: int):
        edges, map_u, map_v = V.build_picross_graph(n)
        return edges, map_u, map_v, V.build_picross_cell_indices(n)

    def prepare(self, batch):
        inputs = torch.as_tensor(np.asarray(batch["inputs"]), dtype=torch.long)
        labels = torch.as_tensor(np.asarray(batch["labels"]), dtype=torch.long)
        B = inputs.shape[0]
        n = int(batch.get("height") or round(math.sqrt(labels.shape[1])))
        edges, map_u, map_v, cell_ids = self._graph(n)
        grid = labels.reshape(B, n, n)
        fwd = {
            "patches": V.picross_patches(inputs, n, self.clue_encoding),  # [2n, B, n, F]
            "edge_indices": edges,
            "model_kwargs": {"map_u": map_u, "map_v": map_v, "cell_ids": cell_ids},
        }
        targets = {
            "slice_labels": V.picross_slice_batch(grid[..., None])[..., 0].permute(1, 0, 2),
            "labels_grid": grid,
            "clues": V.unpack_picross_inputs(inputs, n),  # [B, 2n, L]
        }
        return fwd, targets, {}

    def loss(self, logits_window, targets):
        labels = targets["slice_labels"]  # [2n, B, n]
        return _ce(logits_window, labels.unsqueeze(0).expand(logits_window.shape[:-1])).mean()

    def evaluate(self, logits_final, targets, aux):
        """``logits_final`` ``[2n, B, n, 2]``. Metrics:

        * ``cell_acc``: cells right, after averaging the row and column views of each cell;
        * ``solved``: whole board right;
        * ``line_ok``: fraction of the ``2n`` line clues the predicted board satisfies;
        * ``clue_ok``: boards satisfying every clue (equals ``solved`` when solutions are unique,
          as in ``build_picross``; differs on datasets that keep ambiguous boards);
        * ``view_agree``: cells whose row view and column view predict the same value (how far
          the agents reached a consensus, independently of being right).
        """
        views = logits_final.permute(1, 0, 2, 3)  # [B, 2n, n, 2]
        n = views.shape[2]
        pred = V.reassemble_picross_logits(views).argmax(-1)  # [B, n, n]
        labels = targets["labels_grid"]
        line_ok = (
            picross_line_clues(V.picross_slice_batch(pred[..., None])[..., 0]) == targets["clues"]
        ).all(-1)  # [B, 2n]
        view_pred = views.argmax(-1)
        return {
            "cell_acc": (pred == labels).float().mean().item(),
            "solved": (pred == labels).all(dim=(1, 2)).float().mean().item(),
            "line_ok": line_ok.float().mean().item(),
            "clue_ok": line_ok.all(-1).float().mean().item(),
            "view_agree": (view_pred[:, :n] == view_pred[:, n:].transpose(1, 2))
            .float()
            .mean()
            .item(),
        }


def make_task(name: str, **kwargs):
    tasks = {"maze": MazeTask, "mnist": MNISTTask, "sudoku": SudokuTask, "picross": PicrossTask}
    if name not in tasks:
        raise KeyError(f"unknown task {name!r}; available: {sorted(tasks)}")
    return tasks[name](**kwargs)
