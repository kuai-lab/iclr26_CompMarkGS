# !/bin/bash
echo GPU $1

ROOT_PATH=/home/vision # Adjust this to your project root path
DECODER_CFG_PATH=./decoder/
WM_LOSS=bce
appearance_dim=0
BITS=(48) # 32 48 64

LAMBDA_WM_LOSS=(0.45)

SEEDS=(42)


# Scene list
scene_list=(chair drums ficus hotdog lego materials mic ship)

DATASET=nerf_synthetic


for SEED in "${SEEDS[@]}"; do
for BIT in "${BITS[@]}"; do
    for lambda_wm in "${LAMBDA_WM_LOSS[@]}"; do
        for scene in "${scene_list[@]}"; do
            echo "Running training for scene: $scene"
            CUDA_VISIBLE_DEVICES=$1 python embed_watermark.py  \
                -s ${ROOT_PATH}/Datasets/${DATASET}/$scene \
                --eval \
                --seed ${SEED} \
                --lod 0 \
                --voxel_size 0.001 \
                --decoder_att ${DECODER_CFG_PATH}cfg_${BIT}_${WM_LOSS}.json \
                --update_init_factor 4 \
                --iterations 30_000 \
                --lambda_wm ${lambda_wm} \
                -m outputs/${BIT}bits/seed_${SEED}/${DATASET}/$scene/$lambda_wm  \
                --appearance_dim ${appearance_dim}
        done
    done
done
done


# Scene list
scene_list=(bonsai counter flowers stump treehill bicycle garden kitchen room)

DATASET=MipNeRF360


for SEED in "${SEEDS[@]}"; do
for BIT in "${BITS[@]}"; do
    for lambda_wm in "${LAMBDA_WM_LOSS[@]}"; do
        for scene in "${scene_list[@]}"; do
            echo "Running training for scene: $scene"
            CUDA_VISIBLE_DEVICES=$1 python embed_watermark.py  \
                -s ${ROOT_PATH}/Datasets/${DATASET}/$scene \
                --eval \
                --seed ${SEED} \
                --lod 0 \
                --voxel_size 0.001 \
                --decoder_att ${DECODER_CFG_PATH}cfg_${BIT}_${WM_LOSS}.json \
                --update_init_factor 16 \
                --iterations 30_000 \
                --lambda_wm ${lambda_wm} \
                -m outputs/${BIT}bits/seed_${SEED}/${DATASET}/$scene/$lambda_wm  \
                --appearance_dim ${appearance_dim}
        done
    done
done
done


# Scene list
scene_list=(orchids fern flower fortress horns leaves trex room)

DATASET=nerf_llff_data

for SEED in "${SEEDS[@]}"; do
for BIT in "${BITS[@]}"; do
    for lambda_wm in "${LAMBDA_WM_LOSS[@]}"; do
        for scene in "${scene_list[@]}"; do
            echo "Running training for scene: $scene"
            CUDA_VISIBLE_DEVICES=$1 python embed_watermark.py  \
                -s ${ROOT_PATH}/Datasets/${DATASET}/$scene \
                --eval \
                --seed ${SEED} \
                --lod 0 \
                --voxel_size 0.001 \
                --decoder_att ${DECODER_CFG_PATH}cfg_${BIT}_${WM_LOSS}.json \
                --update_init_factor 16 \
                --iterations 30_000 \
                --lambda_wm ${lambda_wm} \
                -m outputs/${BIT}bits/seed_${SEED}/${DATASET}/$scene/$lambda_wm  \
                --appearance_dim ${appearance_dim}
        done
    done
done
done
