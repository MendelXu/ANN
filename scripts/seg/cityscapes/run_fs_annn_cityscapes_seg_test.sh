#!/usr/bin/env bash

nvidia-smi
PYTHON="python"
WORKDIR=$(cd $(dirname $0)/../../../;pwd)
export PYTHONPATH=${WORKDIR}:${PYTHONPATH}

cd ${WORKDIR}

DATASET="cityscapes"

DATA_DIR=`cat data.conf|grep ${DATASET}|cut -d : -f 2`

BACKBONE="deepbase_resnet101_dilated8"
MODEL_NAME="annn"
CHECKPOINTS_NAME="fs_${MODEL_NAME}_${DATASET}_seg"$2
PRETRAINED_MODEL="./pretrained_models/3x3resnet101-imagenet.pth"

HYPES_FILE="hypes/seg/${DATASET}/fs_annn_${DATASET}_seg_test.json"
MAX_ITERS=120000
LOSS_TYPE="fs_auxohemce_loss"

LOG_DIR="${WORKDIR}/log/seg/${DATASET}/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi


if [ "$1"x == "train"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y --include_val y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 4 5 6 7 --log_to_file n \
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} --norm_type encsync_batchnorm \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} 2>&1 | tee ${LOG_FILE}

elif [ "$1"x == "resume"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --drop_last y --phase train --gathered n --loss_balance y --include_val y \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --gpu 0 1 2 3 4 5 6 7 --log_to_file n\
                       --data_dir ${DATA_DIR} --loss_type ${LOSS_TYPE} --max_iters ${MAX_ITERS} \
                       --resume_continue y --resume ./checkpoints/seg/${DATASET}/${CHECKPOINTS_NAME}_latest.pth \
                       --checkpoints_name ${CHECKPOINTS_NAME} --pretrained ${PRETRAINED_MODEL} --resume_val n 2>&1 | tee -a ${LOG_FILE}

elif [ "$1"x == "debug"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE}--phase debug --gpu 0 --log_to_file n  2>&1 | tee ${LOG_FILE}

elif [ "$1"x == "val"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --gpu 4 5 6 7 --log_to_file n --gathered n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/${DATASET}/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir ${DATA_DIR}/val/image --out_dir val 2>&1 | tee -a ${LOG_FILE}

  ${PYTHON} -u metrics/seg/${DATASET}_evaluator.py --pred_dir results/seg/${DATASET}/${CHECKPOINTS_NAME}/val/label \
                                       --gt_dir ${DATA_DIR}/val/label  2>&1 | tee -a ${LOG_FILE}

elif [ "$1"x == "test"x ]; then
  ${PYTHON} -u main.py --hypes ${HYPES_FILE} --phase test --gpu 0 1 2 3 4 5 6 7 --log_to_file n --gathered n \
                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                       --resume ./checkpoints/seg/${DATASET}/${CHECKPOINTS_NAME}_latest.pth \
                       --test_dir /share/${DATASET}/leftImg8bit/test --out_dir test 2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi
