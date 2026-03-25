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

import os
import numpy as np

import torch
from pathlib import Path
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import prefilter_voxel, render
from decoder.init_decoder import DecoderAttributes
from pytorch_wavelets import DWTForward
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import random

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def bit_acc(decoded, keys):
    diff = (~torch.logical_xor(decoded>0, keys>0))
    bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1]
    return bit_accs
    
def high_pass_filter_dft(image, cutoff=30, sigma=5):
    f = torch.fft.fft2(image)
    fshift = torch.fft.fftshift(f)
    _, C, H, W = image.shape

    crow, ccol = H // 2, W // 2

    x = torch.arange(W, device=image.device).float()
    y = torch.arange(H, device=image.device).float()
    X, Y = torch.meshgrid(y, x, indexing="ij")

    D = torch.sqrt((X - crow) ** 2 + (Y - ccol) ** 2)
    mask = 1 - torch.exp(-((D - cutoff) ** 2) / (2 * sigma ** 2))

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand_as(fshift)

    fshift_filtered = fshift * mask

    f_ishift = torch.fft.ifftshift(fshift_filtered)
    img_back = torch.fft.ifft2(f_ishift)
    img_back = torch.abs(img_back)

    return img_back
    

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, logger=None, ply_path=None, high_success_threshold=6e-1, high_grad_threshold = 15e-5,high_min_opacity=15e-2):
    first_iter = 0
    prepare_output_and_logger(dataset)

    gaussians = GaussianModel(
        dataset.feat_dim, 
        dataset.n_offsets, 
        dataset.voxel_size, 
        dataset.update_depth, 
        dataset.update_init_factor, 
        dataset.update_hierachy_factor, 
        dataset.use_feat_bank, 
        
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
        dataset.qdl_q0, dataset.use_qdl)
    scene = Scene(dataset, gaussians, ply_path=ply_path, shuffle=False)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_wm_loss_for_log = 0.0
    ema_wm_acc_for_log = 0.0
    ema_psnr_wm_for_log = 0.0
    ema_ssim_wm_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    dec_attrs = DecoderAttributes(args.decoder_att, args.seed)
    msg_decoder = dec_attrs.dec
    msg_decoder.eval()
    
    loss_type = dec_attrs.loss_dict
    gt_msg = dec_attrs.msg
    dwt_forward = DWTForward(wave='haar', J=1, mode='symmetric').to('cuda')

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, step=iteration)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, _, scaling, opacity, means2D = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"], render_pkg["means2D"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        wm_loss, wm_acc = 0.0, 0.0
        wm_psnr = psnr(image.detach(), gt_image.detach()).mean()
        wm_ssim = ssim(image, gt_image)

        ssim_loss = (1.0 - ssim(image, gt_image))
        scaling_reg = scaling.prod(dim=1).mean()
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01 * scaling_reg

        if iteration >= 10000:
            gray_image = tf.rgb_to_grayscale(image, num_output_channels=1)
            gray_image_gt = tf.rgb_to_grayscale(gt_image, num_output_channels=1) 
            gray_image = tf.resize(gray_image, (1024, 1024))
            gray_image_gt = tf.resize(gray_image_gt, (1024, 1024))
            
            high_freq = high_pass_filter_dft(gray_image.unsqueeze(0))
            high_freq_gt = high_pass_filter_dft(gray_image_gt.unsqueeze(0))
            high_freq_map = (1.0 - ssim(high_freq, high_freq_gt, size_average=False))
            
            loss = loss + 0.1 * high_freq_map.mean()

        if iteration > 20000:
            LL_img, _ = dwt_forward(image.unsqueeze(0))
            decoded = msg_decoder(LL_img)

            wm_loss = loss_type['loss_w'](decoded, gt_msg)
            wm_acc = bit_acc(decoded, gt_msg).item()
            wm_ramp = min(1.0, max(0.0, (iteration - 20000) / float(max(1, args.wm_ramp_iters))))

            r, g, b = image[0], image[1], image[2]

            max_rgb, _ = torch.max(image, dim=0)
            min_rgb, _ = torch.min(image, dim=0)
            delta = max_rgb - min_rgb

            hue = torch.zeros_like(max_rgb)
            hue[min_rgb == max_rgb] = 0
            hue[(max_rgb == r) & (delta != 0)] = (60 * ((g - b) / delta) % 6)[(max_rgb == r) & (delta != 0)]
            hue[(max_rgb == g) & (delta != 0)] = (60 * ((b - r) / delta) + 120)[(max_rgb == g) & (delta != 0)]
            hue[(max_rgb == b) & (delta != 0)] = (60 * ((r - g) / delta) + 240)[(max_rgb == b) & (delta != 0)]

            saturation = torch.zeros_like(max_rgb)
            saturation[max_rgb != 0] = (delta / max_rgb)[max_rgb != 0]
            value = max_rgb

            hue_ranges = {
                'red': [(0, 60), (300, 360)],
                'green': [(60, 180)],
                'blue': [(180, 300)],
            }

            hue_losses = []
            for color, ranges in hue_ranges.items():
                masks = [(hue >= low) & (hue <= high) for low, high in ranges]
                color_mask = torch.any(torch.stack(masks), dim=0)
                color_mask = color_mask & (saturation >= 0.2) & (value >= 0.2)
                color_mask_tensor = color_mask.unsqueeze(0).unsqueeze(0).float()
                color_loss = F.mse_loss(image.unsqueeze(0) * color_mask_tensor, gt_image.unsqueeze(0) * color_mask_tensor)
                hue_losses.append(color_loss)
                    
            hue_loss = torch.stack(hue_losses).mean()
            
            loss = 10 * (loss + 0.6 * hue_loss) + wm_loss * args.lambda_wm * wm_ramp

        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_psnr_wm_for_log = 0.4 * wm_psnr + 0.6 * ema_psnr_wm_for_log
            ema_ssim_wm_for_log = 0.4 * wm_ssim + 0.6 * ema_ssim_wm_for_log

            if iteration > 20000:
                ema_wm_loss_for_log = 0.4 * wm_loss.item() + 0.6 * ema_wm_loss_for_log
                ema_wm_acc_for_log = 0.4 * wm_acc + 0.6 * ema_wm_acc_for_log

            if iteration % 10 == 0:
                if iteration > 20000:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "WM_Loss": f"{ema_wm_loss_for_log:.{4}f}", "PSNR" : f"{ema_psnr_wm_for_log:.{3}f}", "SSIM" : f"{ema_ssim_wm_for_log:.{3}f}", "BitAcc": f"{ema_wm_acc_for_log:.{3}f}"})
                else:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{4}f}", "PSNR" : f"{ema_psnr_wm_for_log:.{3}f}", "SSIM" : f"{ema_ssim_wm_for_log:.{3}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                if iteration >= 10000:
                    middle = torch.median(high_freq_map)
                    map_mask = (high_freq_map >= middle+0.1) & (high_freq_map <= middle+0.3)
                    high_freq_xy = torch.nonzero(map_mask, as_tuple=False)[:, -2:] # [22626, 2]
                    
                    means2D_np = means2D.detach().cpu().numpy()[:-128]
                    means2D_np_float = np.frombuffer(means2D_np, dtype=np.float32).reshape(-1, 2)
                    means2D_tensor = torch.from_numpy(means2D_np_float).to('cuda')
                    pixel_coords = torch.round(means2D_tensor).to(torch.int32)
                    inside_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < gray_image.shape[2]) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < gray_image.shape[1])
                    
                    hash_pixel = pixel_coords[:, 0] * gray_image.shape[2] + pixel_coords[:, 1]  # shape: (N,)

                    hash_high = high_freq_xy[:, 0] * gray_image.shape[2] + high_freq_xy[:, 1]
                    high_freq_filter = torch.isin(hash_pixel, hash_high)

                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, inside_mask=inside_mask, high_freq_filter=high_freq_filter)
                else:
                    gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                if iteration not in range(3000, 4000):  # let the model get fit to quantization
                    # densification
                    if iteration >= 10000 and iteration % opt.update_interval  == 0:
                            gaussians.adjust_high_anchor(check_interval=opt.update_interval, success_threshold=high_success_threshold, grad_threshold=high_grad_threshold, min_opacity=high_min_opacity)
                    elif iteration < 10000 and iteration % opt.update_interval == 0:
                        gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
                    
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 30_000])
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder_att", type=str, default='-1')
    parser.add_argument("--lambda_wm", type=float, default=0.1)
    parser.add_argument("--wm_ramp_iters", type=int, default=5000)
    parser.add_argument("--high_success_threshold", type=float, default=0.6)
    parser.add_argument("--high_grad_threshold", type=float, default=0.00015)
    parser.add_argument("--high_min_opacity", type=float, default=0.15)
    parser.set_defaults(densify_grad_threshold=0.0007)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    seed_everything(args.seed)
    # enable logging
    
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')
    logger.info(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")

        
    source_path = Path(args.source_path)
    dataset = source_path.name if source_path.name else "unknown_data"
    
    logger.info("Optimizing " + args.model_path)

    # Initialize stdout wrapper state
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training
    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger, high_success_threshold=args.high_success_threshold, high_grad_threshold = args.high_grad_threshold,high_min_opacity=args.high_min_opacity)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, logger=logger, ply_path=new_ply_path, high_success_threshold=args.high_success_threshold, high_grad_threshold = args.high_grad_threshold,high_min_opacity=args.high_min_opacity)

    # All done
    logger.info("\nTraining complete.")
    logger.info("Run eval_compmarkgs.py for rendering and evaluation.")
