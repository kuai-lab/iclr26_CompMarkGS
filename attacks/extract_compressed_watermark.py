#!/usr/bin/env python3

import json
import logging
import os
import sys
import time
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path
from typing import List, Optional

import torch
import torchvision
import torchvision.transforms.functional as tf
from pytorch_wavelets import DWTForward
from PIL import Image
from tqdm import tqdm


def _find_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "decoder").exists() and (candidate / "attacks").exists() and (candidate / "utils").exists():
            return candidate
    return start.parent


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = _find_root(SCRIPT_DIR)
DEFAULT_COMPRESSOR_ROOT = ROOT / "attacks" / "ContextGS"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load LPIPS from the root project first.
from lpipsPyTorch import lpips

if str(DEFAULT_COMPRESSOR_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_COMPRESSOR_ROOT))

from utils.state_utils import build_state, run_codec
from arguments import ModelParams, PipelineParams
from decoder.init_decoder import DecoderAttributes
from gaussian_renderer import prefilter_voxel, render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.image_utils import psnr
from utils.loss_utils import ssim


def get_logger(path: str) -> logging.Logger:
    logger = logging.getLogger("eval_compmarkgs_after_compression")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    os.makedirs(path, exist_ok=True)
    fileinfo = logging.FileHandler(os.path.join(path, "compressed_eval_outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)
    return logger



def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs


def read_image(renders_dir: Path, gt_dir: Path):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        render_img = Image.open(renders_dir / fname)
        gt_img = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render_img).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    if len(views) == 0:
        return [], []

    t_list: List[float] = []
    visible_count_list = []
    per_view_dict = {}

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        with torch.no_grad():
            torch.cuda.synchronize(); t0 = time.time()
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
            torch.cuda.synchronize(); t1 = time.time()
            torch.cuda.empty_cache()
            t_list.append(t1 - t0)

            rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
            visible_count = (render_pkg["radii"] > 0).sum()
            visible_count_list.append(visible_count.item())
            gt = view.original_image[0:3, :, :]

            errormap = (rendering - gt).abs()
            out_name = "{0:05d}.png".format(idx)
            torchvision.utils.save_image(rendering, os.path.join(render_path, out_name))
            torchvision.utils.save_image(errormap, os.path.join(error_path, out_name))
            torchvision.utils.save_image(gt, os.path.join(gts_path, out_name))
            per_view_dict[out_name] = float(visible_count.item())

        del voxel_visible_mask, render_pkg, rendering, gt, errormap

    per_view_count_path = os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json")
    with open(per_view_count_path, "w") as fp:
        json.dump(per_view_dict, fp, indent=2)

    return t_list, visible_count_list


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    *,
    skip_train: bool = True,
    skip_test: bool = False,
    args,
    runtime,
    logger: logging.Logger,
):
    gaussians = GaussianModel(
        dataset.feat_dim,
        dataset.n_offsets,
        dataset.voxel_size,
        dataset.update_depth,
        dataset.update_init_factor,
        dataset.update_hierachy_factor,
        dataset.use_feat_bank,
        n_features_per_level=args.n_features,
        decoded_version=False,
        level_num=args.level_num,
        hyper_divisor=args.hyper_divisor,
        target_ratio=args.target_ratio,
    )

    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    gaussians.eval()

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    codec_dir = Path(dataset.model_path) / "bitstreams"
    gaussians.decoded_version = False
    anchor_device = gaussians._anchor.device
    if gaussians.x_bound_min.device != anchor_device:
        gaussians.x_bound_min = gaussians.x_bound_min.to(anchor_device)
    if gaussians.x_bound_max.device != anchor_device:
        gaussians.x_bound_max = gaussians.x_bound_max.to(anchor_device)
    codec_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[Stage] Codec encode/decode")
    with run_codec():
        log_info = gaussians.estimate_final_bits()
        logger.info(log_info)
        log_info = gaussians.conduct_encoding(pre_path_name=str(codec_dir))
        logger.info(log_info)
        log_info = gaussians.conduct_decoding(pre_path_name=str(codec_dir))
        logger.info(log_info)
    torch.cuda.empty_cache()

    visible_count = None
    if not skip_train:
        render_set(
            dataset.model_path,
            "train_post",
            scene.loaded_iter,
            scene.getTrainCameras(),
            gaussians,
            pipeline,
            background,
        )

    if not skip_test:
        _, visible_count = render_set(
            dataset.model_path,
            "test_post",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
        )

    runtime.post_visible_count = visible_count
    runtime.post_loaded_iter = scene.loaded_iter

    return visible_count


def evaluate(
    model_path: str,
    split: str,
    decoder_att: str,
    seed: int,
    logger: Optional[logging.Logger] = None,
):
    scene_dir = Path(model_path)
    split_dir = scene_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    dec_attrs = DecoderAttributes(cfg_path=decoder_att, seed=seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder = dec_attrs.dec
    decoder.eval()
    dwt_forward = DWTForward(wave="haar", J=1, mode="symmetric").to(device)

    with open(dec_attrs.message_path, "r") as f:
        key = f.read().strip()
    keyarr = [int(v) for v in key]
    keytensor = torch.tensor(keyarr, dtype=torch.float32, device=device)[None, :]

    full_dict = {}
    per_view_dict = {}
    full_dict[str(scene_dir)] = {}
    per_view_dict[str(scene_dir)] = {}

    method_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
    if len(method_dirs) == 0:
        raise RuntimeError(f"No rendered methods found under {split_dir}")

    for method_dir in method_dirs:
        method = method_dir.name
        gt_dir = method_dir / "gt"
        renders_dir = method_dir / "renders"
        if not gt_dir.exists() or not renders_dir.exists():
            continue

        renders, gts, image_names = read_image(renders_dir, gt_dir)
        per_view_counts = {}
        per_view_count_path = method_dir / "per_view_count.json"
        if per_view_count_path.exists():
            with per_view_count_path.open("r") as fp:
                per_view_counts = json.load(fp)

        ssims = []
        psnrs = []
        lpipss = []
        bit_accs = []
        for idx in tqdm(range(len(renders)), desc=f"Metric eval [{method}]"):
            with torch.no_grad():
                ll_img, _ = dwt_forward(renders[idx])
                pred_msg = decoder(ll_img)
                bit_accs.append(float(bit_acc(keytensor, pred_msg).mean().item()))

            ssims.append(float(ssim(renders[idx], gts[idx]).item()))
            psnrs.append(float(psnr(renders[idx], gts[idx]).mean().item()))
            lpipss.append(float(lpips(renders[idx], gts[idx], net_type="vgg").item()))

        mean_ssim = float(torch.tensor(ssims).mean().item()) if ssims else None
        mean_psnr = float(torch.tensor(psnrs).mean().item()) if psnrs else None
        mean_lpips = float(torch.tensor(lpipss).mean().item()) if lpipss else None
        mean_bit_acc = float(torch.tensor(bit_accs).mean().item()) if bit_accs else None

        if logger is not None:
            logger.info(f"model_path: {scene_dir}")
            logger.info(f"  split  : {split}")
            logger.info(f"  SSIM   : {mean_ssim}")
            logger.info(f"  PSNR   : {mean_psnr}")
            logger.info(f"  LPIPS  : {mean_lpips}")
            logger.info(f"  bit_acc: {mean_bit_acc}")

        full_dict[str(scene_dir)][method] = {
            "SSIM": mean_ssim,
            "PSNR": mean_psnr,
            "LPIPS": mean_lpips,
            "bit_accs": mean_bit_acc,
        }
        per_view_dict[str(scene_dir)][method] = {
            "SSIM": {name: value for name, value in zip(image_names, ssims)},
            "PSNR": {name: value for name, value in zip(image_names, psnrs)},
            "LPIPS": {name: value for name, value in zip(image_names, lpipss)},
            "bit_accs": {name: value for name, value in zip(image_names, bit_accs)},
            "VISIBLE_COUNT": {name: per_view_counts.get(name) for name in image_names},
        }

    method_metrics = next(iter(full_dict[str(scene_dir)].values()), None)

    results_json_path = scene_dir / "results_compressed.json"
    result_payload = {
        "results_post": method_metrics,
    }
    with results_json_path.open("w") as fp:
        json.dump(result_payload, fp, indent=2)

    if logger is not None:
        logger.info(f"Saved: {results_json_path}")

    return method_metrics


def main():
    parser = ArgumentParser(description="CompMarkGS render & evaluation script after compression")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compressor_checkpoint_path", type=str, required=True)
    parser.add_argument("--level_num", type=int, default=3)
    parser.add_argument("--n_features", type=int, default=4)
    parser.add_argument("--decoder_att", type=str, required=True)
    parser.add_argument("--skip_train", type=int, default=1)
    parser.add_argument("--skip_test", type=int, default=0)
    args = parser.parse_args()

    logger = get_logger(args.model_path)
    logger.info(f"args: {args}")

    runtime = build_state(args=args)
    dataset = lp.extract(args)
    pipeline = pp.extract(args)

    logger.info("Starting rendering stage...")
    render_sets(
        dataset,
        runtime.out_iteration,
        pipeline,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        args=args,
        runtime=runtime,
        logger=logger,
    )

    logger.info("Starting evaluation on split=test...")
    evaluate(
        model_path=str(runtime.out_model_path),
        split="test_post",
        decoder_att=args.decoder_att,
        seed=args.seed,
        logger=logger,
    )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
