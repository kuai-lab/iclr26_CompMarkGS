# CompMarkGS Docker Image: Loading and Running

The Docker image already contains the **code, Python environment, model weights, and 25 trained checkpoints**, so **no build step is required**.

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

docker load -i image/compmarkgs_48bit.tar

# Verify that the image has been loaded.
# You should see compmarkgs:48bit.
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
bash docker/run.sh train 1234 5678
```

### Options

```bash
# Select specific GPUs.
# By default, all visible GPUs are used.
GPU_IDS=0,1 bash docker/run.sh train 1234

# Run only selected scenes.
# This also applies to infer.
ONLY=nerf_synthetic/lego,llff/fern bash docker/run.sh train 1234

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

Progress logs are written to:

```text
outputs/sched.log
outputs/eval.log
```

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

## Directory Structure

```text
iclr26_compmarkgs_docker/
├── LOAD_AND_RUN.md          # This file
├── image/                   # Docker image archive + SHA256SUMS
├── docker/run.sh            # Container launcher (check / infer / train / shell)
└── dataset/                 # LLFF (8), mip-NeRF 360 (9), NeRF Synthetic (8), 15 GB
```