#!/bin/bash
DATASET=$1
SEED=$2
N_CTX=$3
CUDA_DEVICE_ID=$4

export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID}

# cd ../..

# custom config
DATA=./data
TRAINER=MM_CLS_OP
# TRAINER=CoOp



CFG=vit_b16_c4_ep50_imagenet21k_pretrain
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=64


DIR=output_ovmr/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --n_ctx ${N_CTX} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES all
fi