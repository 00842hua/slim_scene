#!/bin/bash
TF_DIR=/home/work/wangfei11/data/video_labeling/TF_136_20190619
CKPT_DIR=/home/work/yangsheng/res/0630_mixup_alpha_05_percent_1
PRETRAIN=/home/work/wangfei11/PRE_TRAINED/mobilenet_v2_1.0_224.ckpt
label_count=136
image_count=570000
MAX_STEPS=1000000
BATCH_SIZE=128
QUANT_DELAY=-1

export CUDA_VISIBLE_DEVICES=4

python3 -u train_image_classifier.py \
 --num_clones=1 \
 --train_dir=${CKPT_DIR} \
 --dataset_name=default \
 --tf_file_pattern=videolabel_%s_*.tfrecord \
 --dataset_dir=${TF_DIR} \
 --num_readers=16 \
 --num_preprocessing_threads=16 \
 --label_count=${label_count} \
 --train_image_count=${image_count} \
 --dataset_split_name=train \
 --rotate_image=False \
 --model_name=mobilenet_v2 \
 --preprocessing_name=inception_common \
 --crop_area_min=0.8 \
 --batch_size=${BATCH_SIZE} \
 --learning_rate=0.001 \
 --learning_rate_decay_factor=0.94 \
 --save_interval_secs=1800 \
 --checkpoint_path=${PRETRAIN} \
 --checkpoint_exclude_scopes='MobilenetV2/Logits,global_step' \
 --max_number_of_steps=${MAX_STEPS} \
 --train_image_size=224 \
 --moving_average_decay=0.9999 \
 --quantize_delay=${QUANT_DELAY} \
 --ignore_missing_vars=True \
 --mixup=True \
 --mixup_alpha=0.5 \
 --mixup_percent=1 \
 --weight_decay=0.00004 \

