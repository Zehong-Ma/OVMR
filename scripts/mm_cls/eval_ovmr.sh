#!/bin/bash
DATASET=$1
SEED=$2
SUB_CLASSES=$3
N_CTX=$4
EVAL_MODE=$5
EVAL_TAU=$6
CUDA_DEVICE_ID=$7
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICE_ID}
# cd ../..

# custom config
DATA=./data
TRAINER=MM_CLS_OP
# TRAINER=CoOp



CFG=vit_b16_c4_ep50_imagenet21k_pretrain
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=16
LOADEP=30
SUB=${SUB_CLASSES}


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
COMMON_DIR_train=imagenet_21k_P/shots_64/${TRAINER}/${CFG}/seed1
MODEL_DIR=output_ovmr/base2new/train_base/${COMMON_DIR_train}
DIR=output_ovmr/base2new/test_${SUB}_${EVAL_MODE}_tau${EVAL_TAU}/${COMMON_DIR}
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
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval_mode ${EVAL_MODE} \
    --eval_tau ${EVAL_TAU} \
    --n_ctx ${N_CTX} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi