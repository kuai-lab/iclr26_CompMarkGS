# CompMarkGS Docker Image: Loading and Running

The Docker image already contains the **code, Python environment, model weights, 25 trained checkpoints, and ContextGS for compressed extraction**, so **no build step is required**.

## 0. Requirements

- x86_64 (amd64) Linux host
- **NVIDIA driver 570 or later**
- NVIDIA Container Toolkit configured so that `docker run --gpus all ...` works
- Supported GPUs: sm_80–sm_120  
  (A100, A6000, RTX 30/40 series, L40, RTX 6000 Ada, H100, B200, RTX PRO 6000 Blackwell, RTX 50 series)
- `check` and `infer` have been verified on an **RTX PRO 6000 Blackwell (sm_120)**
- Disk space:
  - Approximately 35 GB for the loaded Docker image
  - Approximately 29 GB for this directory
  - Additional space for outputs
- VRAM:
  - Evaluation requires relatively little VRAM
  - Training may use up to approximately **34 GB per scene** (`LLFF/leaves`)
  - A GPU with **40 GB or more VRAM** is recommended

## 1. Load the Image

This only needs to be done once.

```bash
cd iclr26_compmarkgs_docker

# Optional: verify that the transferred files are not corrupted
(cd image && sha256sum -c SHA256SUMS)

docker load -i image/compmarkgs_48bit_compressed.tar

# Verify that the image has been loaded.
# You should see compmarkgs:48bit-compressed.
docker images compmarkgs
```

## 2. Run

Run the following commands from this directory.

```bash
# Check GPU compatibility.
# The setup is working correctly if the last line prints ALL_OK.
bash docker/run.sh check

# Evaluate the 25 checkpoints included in the image without training.
# This performs rendering, decoding, and PSNR/SSIM/LPIPS evaluation.
bash docker/run.sh infer

# Training + evaluation.
# Provide only the message seed.
# All 25 scenes are trained and automatically evaluated afterward.
bash docker/run.sh train 57494

# Multiple seeds can be provided.
# Each seed is processed sequentially.
# Seeds other than 57494 embed a different message, so their results differ from the table below.
bash docker/run.sh train 1234 5678

# Compressed extraction (ContextGS, = scripts/extract_compressed_watermark.sh).
# Compresses the 25 included checkpoints and evaluates bit accuracy / PSNR / SSIM / LPIPS,
# then prints a summary table (per scene, dataset means, difference to the stored reference values).
# The argument is the host GPU (default: 0); the 25 scenes run one after another on it.
bash docker/run.sh extract_compressed 0

# With a comma-separated list of host GPUs, the 25 scenes are split across them
# (one scene per GPU at a time; a GPU takes the next scene as soon as it finishes one).
bash docker/run.sh extract_compressed 4,5,6,7
```

### Seeds and Expected Results

- **Model seed: 42.** Fixed in `train` (the official `--seed 42`); it is not a run argument.
- **Message seed of the included checkpoints: 57494.** Recorded in `checkpoints/48bit/INFO.json` inside the image.
  `infer` and `extract_compressed` read it from there, so they take no seed argument.
- To reproduce the numbers below by training, use message seed **57494**: `bash docker/run.sh train 57494`.

Expected results with message seed 57494 (mean over scenes):

| Command | Dataset | PSNR | SSIM | LPIPS | Bit Acc. (%) |
|---|---|---:|---:|---:|---:|
| `infer` / `train 57494` | NeRF Synthetic (8) | 31.89 | 0.963 | 0.041 | 92.54 |
| | LLFF (8) | 25.36 | 0.800 | 0.206 | 99.26 |
| | Mip-NeRF 360 (9) | 27.43 | 0.822 | 0.212 | 98.34 |
| | **All (25)** | **28.19** | **0.860** | **0.155** | **96.78** |
| `extract_compressed` | NeRF Synthetic (8) | 31.08 | 0.955 | 0.049 | 91.17 |
| | LLFF (8) | 24.89 | 0.785 | 0.223 | 99.02 |
| | Mip-NeRF 360 (9) | 26.37 | 0.778 | 0.231 | 97.83 |
| | **All (25)** | **27.40** | **0.837** | **0.170** | **96.08** |

`infer` reproduces these numbers exactly. The `infer` and `extract_compressed` summaries both print the difference to the stored per-scene reference values.
Training is not bit-exact, so a new `train 57494` run differs slightly per scene (e.g. `nerf_synthetic/lego`: 91.60% vs 91.72% bit accuracy).

### Options

```bash
# Select specific GPUs.
# By default, all visible GPUs are used (extract_compressed: GPU 0).
GPU_IDS=0,1 bash docker/run.sh train 57494
GPU_IDS=4,5,6,7 bash docker/run.sh extract_compressed   # same as: extract_compressed 4,5,6,7

# Run only selected scenes, as <dataset>/<scene> with dataset nerf_synthetic, nerf_llff_data or MipNeRF360.
# This also applies to infer and extract_compressed.
ONLY=nerf_synthetic/lego,nerf_llff_data/fern bash docker/run.sh train 57494
ONLY=nerf_synthetic/lego,nerf_llff_data/fern bash docker/run.sh extract_compressed 4,5

# Maximum number of concurrent jobs per GPU.
# Default: 1
# Using 2 or more typically makes each job 2–4× slower.
MAX_PER_GPU=1
```

### Output Locations

Training results are stored under:

```text
outputs/48bits/seed_42_msg_<seed>/<dataset>/<scene>/0.45/
```

Each training output directory contains:

```text
point_cloud/iteration_30000/   # trained model
outputs.log                    # training log
results.json                   # evaluation results
test/ours_30000/               # rendered outputs
```

Checkpoint evaluation results are stored under:

```text
outputs/infer/48bit/<dataset>/<scene>/results.json
```

A comparison against the stored reference values is written to:

```text
outputs/infer_summary.txt
```

Overall evaluation summaries are written to:

```text
outputs/eval_summary.txt
outputs/eval_summary.csv
```

Compressed extraction results are stored under:

```text
outputs/compressed/<dataset>/<scene>/compression_codec/results_compressed.json
outputs/compressed/<dataset>/<scene>/extract_compressed.log   # extraction log
```

With a GPU list, the terminal shows one `start` and one `done` / `FAIL` line per scene, with the time and GPU.

The compressed extraction summary is written to:

```text
outputs/compressed_summary.txt   # per scene PSNR / SSIM / LPIPS / bit%, dataset means, difference to the reference
outputs/compressed_summary.csv
```

The table covers every result under `outputs/compressed/`, so after re-running a subset with `ONLY=...` it again shows all 25 scenes.

Progress logs are written to:

```text
outputs/sched.log
outputs/eval.log
```

Note: `extract_compressed` compresses the trained checkpoints afterwards; for a higher compression ratio, train CompMarkGS within the HAC / ContextGS framework as in the paper.

### Dataset and Output Paths

`run.sh` mounts the following directories into the container:

- `dataset/` — mounted as read-only
- `outputs/`

To use a different dataset location:

```bash
DATA=/path/to/dataset bash docker/run.sh infer
```

To use a different output location:

```bash
OUT=/path/to/outputs bash docker/run.sh infer
```

The container runs with the host user's UID/GID, so generated files are owned by the host user rather than root.

## Without Docker

Use the official code with the watermarked checkpoints ([download](CHECKPOINT_LINK), 25 scenes). Pass the message seed (57494) as `--seed`.

```bash
tar -xzf compmarkgs_48bit_watermark_checkpoints_25scenes.tar.gz   # -> checkpoints/48bit/<dataset>/<scene>/
git clone https://github.com/kuai-lab/iclr26_CompMarkGS.git && cd iclr26_CompMarkGS   # environment: see the official README
python extract_watermark.py -m ../checkpoints/48bit/nerf_synthetic/lego -s ../dataset/nerf_synthetic/lego \
    --eval --seed 57494 --decoder_att ./decoder/cfg_48_bce.json --skip_train 1 --skip_test 0
```

## Directory Structure

```text
iclr26_compmarkgs_docker/
├── LOAD_AND_RUN.md          # This file
├── image/                   # Docker image archive + SHA256SUMS
├── docker/run.sh            # Container launcher (check / infer / train / extract_compressed / shell)
└── dataset/                 # LLFF (8), mip-NeRF 360 (9), NeRF Synthetic (8), 15 GB
```
