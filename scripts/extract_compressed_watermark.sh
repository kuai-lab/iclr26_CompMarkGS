#!/bin/bash
echo GPU $1

ROOT_PATH=/home/vision # Adjust this to your project root path

DATASET_NAME=nerf_synthetic # nerf_llff_data, MipNeRF360

DATA_PATH=${ROOT_PATH}/Datasets/${DATASET_NAME}/
MODEL_PATH=${ROOT_PATH}/outputs/${DATASET_NAME}/
DECODER_CFG_PATH=./decoder/

COMPRESSOR_CKPT_ROOT=${ROOT_PATH}/attacks/ContextGS/outputs # Adjust this to your compressor checkpoint root path
COMPRESSOR_ITERATION=30000
COMPRESSOR_SUFFIX=0.004

WM_LOSS=bce
BITS=(48)
SEEDS=(42)
LAMBDA_WM_LOSS=(0.45)

SCENES=(bonsai counter flowers stump treehill bicycle garden kitchen room)
# SCENES=(chair drums ficus hotdog lego materials mic ship)
# SCENES=(orchids fern flower fortress horns leaves trex room)

for seed in "${SEEDS[@]}"; do
    for bit in "${BITS[@]}"; do
        for lambda_wm in "${LAMBDA_WM_LOSS[@]}"; do
            for scene in "${SCENES[@]}"; do
                echo "Running compressed extraction for scene: ${scene}"
                CUDA_VISIBLE_DEVICES=$1 python attacks/extract_compressed_watermark.py \
                    -m "${MODEL_PATH}${bit}bits/seed_${seed}/${scene}/${lambda_wm}" \
                    -s "${DATA_PATH}${scene}" \
                    --eval \
                    --seed "${seed}" \
                    --decoder_att "${DECODER_CFG_PATH}cfg_${bit}_${WM_LOSS}.json" \
                    --compressor_checkpoint_path "${COMPRESSOR_CKPT_ROOT}/${DATASET_NAME}/${scene}_${COMPRESSOR_SUFFIX}/point_cloud/iteration_${COMPRESSOR_ITERATION}/checkpoint.pth"
            done
        done
    done
done
