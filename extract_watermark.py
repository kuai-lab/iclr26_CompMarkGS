#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import json
import logging
import os
import time
from argparse import ArgumentParser
from os import makedirs
from pathlib import Path
from typing import Optional

import torch
import torchvision
import torchvision.transforms.functional as tf
from PIL import Image
from lpipsPyTorch import lpips
from pytorch_wavelets import DWTForward
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from decoder.init_decoder import DecoderAttributes
from gaussian_renderer import prefilter_voxel, render
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim


def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
    return bit_accs


def get_logger(path) -> logging.Logger:
    logger = logging.getLogger("eval_compmarkgs")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(os.path.join(path, "eval_outputs.log"))
    stream_handler = logging.StreamHandler()
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        torch.cuda.synchronize();t_start = time.time()

        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)


        # gts
        gt = view.original_image[0:3, :, :]

        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)

    return t_list, visible_count_list


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool = True,
    skip_test: bool = False,
    logger: Optional[logging.Logger] = None,
):
    with torch.no_grad():
        gaussians = GaussianModel(
            dataset.feat_dim,
            dataset.n_offsets,
            dataset.voxel_size,
            dataset.update_depth,
            dataset.update_init_factor,
            dataset.update_hierachy_factor,
            dataset.use_feat_bank,
            dataset.appearance_dim,
            dataset.ratio,
            dataset.add_opacity_dist,
            dataset.add_cov_dist,
            dataset.add_color_dist,
            dataset.qdl_q0,
            dataset.use_qdl,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            t_train_list, _ = render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
            )
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean().item()
            if logger is not None:
                logger.info(f"Train FPS: {train_fps:.5f}")

        if not skip_test:
            t_test_list, _ = render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
            )
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean().item()
            if logger is not None:
                logger.info(f"Test FPS: {test_fps:.5f}")

        return scene.loaded_iter


def read_images(renders_dir: Path, gt_dir: Path):
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

        renders, gts, image_names = read_images(renders_dir, gt_dir)
        per_view_counts = {}
        per_view_count_path = method_dir / "per_view_count.json"
        if per_view_count_path.exists():
            with open(per_view_count_path, "r") as fp:
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

    with open(scene_dir / "results.json", "w") as fp:
        json.dump(full_dict[str(scene_dir)], fp, indent=2)
    with open(scene_dir / "per_view.json", "w") as fp:
        json.dump(per_view_dict[str(scene_dir)], fp, indent=2)


def main():
    parser = ArgumentParser(description="CompMarkGS render & evaluation script")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--decoder_att", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_train", type=int, default=1)
    parser.add_argument("--skip_test", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)

    os.makedirs(args.model_path, exist_ok=True)
    logger = get_logger(args.model_path)
    logger.info(f"args: {args}")

    safe_state(args.quiet)

    model_args = model.extract(args)
    pipeline_args = pipeline.extract(args)

    logger.info("Starting rendering stage...")
    loaded_iter = render_sets(
        model_args,
        args.iteration,
        pipeline_args,
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        logger=logger,
    )
    logger.info(f"Rendering complete. loaded_iter={loaded_iter}")

    logger.info("Starting evaluation on split=test...")
    evaluate(
        model_path=args.model_path,
        split="test",
        decoder_att=args.decoder_att,
        seed=args.seed,
        logger=logger,
    )
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
