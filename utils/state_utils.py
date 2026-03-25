import io
import shutil
import sys
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _max_iteration(point_cloud_root: Path) -> int:
    max_iter = -1
    for child in point_cloud_root.iterdir():
        if child.is_dir() and child.name.startswith("iteration_"):
            try:
                iter_idx = int(child.name.split("_")[-1])
                max_iter = max(max_iter, iter_idx)
            except ValueError:
                pass
    if max_iter < 0:
        raise RuntimeError(f"No iteration_* directory found under: {point_cloud_root}")
    return max_iter


def _get_iter_dir(
    scaffold_model_path: Path,
    scaffold_point_cloud_dir: str,
    scaffold_iteration: int,
) -> Tuple[Path, int]:
    if scaffold_model_path.name.startswith("iteration_") and (scaffold_model_path / "point_cloud.ply").exists():
        return scaffold_model_path, int(scaffold_model_path.name.split("_")[-1])

    point_cloud_root = scaffold_model_path / scaffold_point_cloud_dir
    if not point_cloud_root.exists():
        raise FileNotFoundError(f"Point cloud dir not found: {point_cloud_root}")

    iter_idx = scaffold_iteration if scaffold_iteration != -1 else _max_iteration(point_cloud_root)
    iter_dir = point_cloud_root / f"iteration_{iter_idx}"
    if not iter_dir.exists():
        raise FileNotFoundError(f"Iteration dir not found: {iter_dir}")
    if not (iter_dir / "point_cloud.ply").exists():
        raise FileNotFoundError(f"Missing point_cloud.ply in {iter_dir}")
    return iter_dir, iter_idx


def build_ply(
    src_ply: Path,
    dst_ply: Path,
    hyper_divisor: int,
    feat_dim: int,
    n_offsets: int,
) -> None:
    ply = PlyData.read(str(src_ply))
    vertex = ply.elements[0]
    props = vertex.properties

    def collect_names(prefix: str):
        names = [p.name for p in props if p.name.startswith(prefix)]
        return sorted(names, key=lambda x: int(x.split("_")[-1]))

    def matrix(names):
        if not names:
            return np.zeros((vertex.count, 0), dtype=np.float32)
        arr = np.zeros((vertex.count, len(names)), dtype=np.float32)
        for i, name in enumerate(names):
            arr[:, i] = np.asarray(vertex[name], dtype=np.float32)
        return arr

    anchor = np.stack(
        [
            np.asarray(vertex["x"], dtype=np.float32),
            np.asarray(vertex["y"], dtype=np.float32),
            np.asarray(vertex["z"], dtype=np.float32),
        ],
        axis=1,
    )
    normals = np.zeros_like(anchor, dtype=np.float32)
    opacity = np.asarray(vertex["opacity"], dtype=np.float32)[..., None]

    offset_names = collect_names("f_offset")
    feat_names = collect_names("f_anchor_feat")
    scale_names = collect_names("scale_")
    rot_names = collect_names("rot")

    if not feat_names:
        raise RuntimeError(f"No f_anchor_feat_* fields in {src_ply}")
    if not offset_names or len(offset_names) % 3 != 0:
        raise RuntimeError(
            f"Invalid f_offset_* fields in {src_ply}, count={len(offset_names)} (expected positive multiple of 3)."
        )
    if len(feat_names) != feat_dim:
        raise RuntimeError(
            f"feature dim mismatch: scaffold has {len(feat_names)} but args.feat_dim={feat_dim}"
        )
    if len(offset_names) != (n_offsets * 3):
        raise RuntimeError(
            f"offset dim mismatch: scaffold has {len(offset_names)} fields "
            f"but args.n_offsets={n_offsets} expects {n_offsets * 3}"
        )

    offsets = matrix(offset_names)
    anchor_feat = matrix(feat_names)
    scales = matrix(scale_names)
    rots = matrix(rot_names)

    hyper_channels = feat_dim // hyper_divisor
    masks = np.ones((anchor.shape[0], n_offsets), dtype=np.float32)
    hyper = np.zeros((anchor.shape[0], hyper_channels), dtype=np.float32)

    attr_names = ["x", "y", "z", "nx", "ny", "nz"]
    attr_names += [f"f_offset_{i}" for i in range(n_offsets * 3)]
    attr_names += [f"f_mask_{i}" for i in range(n_offsets)]
    attr_names += [f"f_anchor_feat_{i}" for i in range(feat_dim)]
    attr_names += [f"f_hyper_latent_{i}" for i in range(hyper.shape[1])]
    attr_names += ["opacity"]
    attr_names += [f"scale_{i}" for i in range(scales.shape[1])]
    attr_names += [f"rot_{i}" for i in range(rots.shape[1])]

    dtype_full = [(name, "f4") for name in attr_names]
    values = np.concatenate((anchor, normals, offsets, masks, anchor_feat, hyper, opacity, scales, rots), axis=1)

    elements = np.empty(anchor.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, values))

    dst_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(dst_ply))


def _fit_state_dict_overlap(dst_state: Dict[str, torch.Tensor], src_state: Dict[str, torch.Tensor]):
    out_state = {}
    for key, dst_tensor in dst_state.items():
        if key not in src_state:
            out_state[key] = dst_tensor
            continue

        src_tensor = src_state[key]
        out_tensor = torch.zeros_like(dst_tensor)
        if dst_tensor.ndim == src_tensor.ndim:
            if dst_tensor.shape == src_tensor.shape:
                out_tensor = src_tensor.clone()
            elif dst_tensor.ndim == 1:
                n0 = min(dst_tensor.shape[0], src_tensor.shape[0])
                out_tensor[:n0] = src_tensor[:n0]
            elif dst_tensor.ndim == 2:
                n0 = min(dst_tensor.shape[0], src_tensor.shape[0])
                n1 = min(dst_tensor.shape[1], src_tensor.shape[1])
                out_tensor[:n0, :n1] = src_tensor[:n0, :n1]
        out_state[key] = out_tensor
    return out_state


def _update_pack(checkpoint: Dict[str, object], scaffold_iter_dir: Path) -> Dict[str, object]:
    out_state = dict(checkpoint)
    src_map = {
        "opacity_mlp": scaffold_iter_dir / "opacity_mlp.pt",
        "cov_mlp": scaffold_iter_dir / "cov_mlp.pt",
        "color_mlp": scaffold_iter_dir / "color_mlp.pt",
    }
    for key, pt_path in src_map.items():
        if not pt_path.exists():
            continue
        module = torch.jit.load(str(pt_path), map_location="cpu")
        src_state = {k: v.detach().cpu() for k, v in module.state_dict().items()}

        if key not in out_state or not isinstance(out_state[key], dict):
            continue
        dst_state = out_state[key]
        out_state[key] = _fit_state_dict_overlap(dst_state, src_state)
    return out_state


def build_state(*, args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    source_model_path = str(args.model_path)

    scaffold_iter_dir, resolved_iter = _get_iter_dir(
        scaffold_model_path=Path(args.model_path),
        scaffold_point_cloud_dir="point_cloud",
        scaffold_iteration=args.iteration,
    )
    scaffold_root = scaffold_iter_dir.parent.parent
    src_ply = scaffold_iter_dir / "point_cloud.ply"

    feat_dim = int(args.feat_dim)
    n_offsets = int(args.n_offsets)
    voxel_size = 0.001 if args.voxel_size is None else float(args.voxel_size)

    out_model_path = Path(args.model_path) / "compression_codec"
    out_iteration = 0
    if out_model_path.exists():
        shutil.rmtree(out_model_path)
    out_iter_dir = out_model_path / "point_cloud" / f"iteration_{out_iteration}"
    out_ply = out_iter_dir / "point_cloud.ply"
    out_ckpt = out_iter_dir / "checkpoint.pth"
    out_iter_dir.mkdir(parents=True, exist_ok=True)

    build_ply(
        src_ply=src_ply,
        dst_ply=out_ply,
        hyper_divisor=args.hyper_divisor,
        feat_dim=feat_dim,
        n_offsets=n_offsets,
    )
    checkpoint = torch.load(args.compressor_checkpoint_path, map_location="cpu")
    checkpoint = _update_pack(checkpoint, scaffold_iter_dir=scaffold_iter_dir)
    torch.save(checkpoint, str(out_ckpt))

    args.model_path = str(out_model_path)

    return SimpleNamespace(
        source_model_path=source_model_path,
        scaffold_iter_dir=scaffold_iter_dir,
        resolved_iter=resolved_iter,
        scaffold_root=scaffold_root,
        src_ply=src_ply,
        voxel_size=voxel_size,
        out_model_path=out_model_path,
        out_iter_dir=out_iter_dir,
        out_ply=out_ply,
        out_ckpt=out_ckpt,
        feat_dim=feat_dim,
        out_iteration=out_iteration,
        n_offsets=n_offsets,
    )


def begin_pre_mode(gaussians):
    state = {
        "decoded_version": gaussians.decoded_version,
        "_anchor": gaussians._anchor,
        "_scaling": gaussians._scaling,
        "_mask": gaussians._mask,
    }
    with torch.no_grad():
        scaling_dec = torch.exp(gaussians._scaling.detach())
        mask_dec = (torch.sigmoid(gaussians._mask.detach()) > 0.01).float()
        anchor_dec = gaussians._anchor.detach().clone()
        gaussians._scaling = torch.nn.Parameter(scaling_dec)
        gaussians._mask = torch.nn.Parameter(mask_dec)
        gaussians._anchor = torch.nn.Parameter(anchor_dec)
        gaussians.decoded_version = True
    return state


def end_pre_mode(gaussians, state):
    gaussians.decoded_version = state["decoded_version"]
    gaussians._anchor = state["_anchor"]
    gaussians._scaling = state["_scaling"]
    gaussians._mask = state["_mask"]


class _StdoutLineFilter(io.TextIOBase):
    def __init__(self, stream, blocked_tokens):
        self.stream = stream
        self.blocked_tokens = blocked_tokens

    def write(self, text):
        if any(token in text for token in self.blocked_tokens):
            return len(text)
        return self.stream.write(text)

    def flush(self):
        self.stream.flush()


@contextmanager
def run_codec() -> Iterator[None]:
    filter_stream = _StdoutLineFilter(
        sys.stdout,
        blocked_tokens=("[Warn] Missing context at", "[Warn] Skip context at"),
    )
    with redirect_stdout(filter_stream):
        yield
