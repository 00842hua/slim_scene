export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PYTHONPATH=$PYTHONPATH:`pwd`

TRAIN_DIR=/home/work/wangfei11/ImgageNetCKPT/ckpt_efficientnet_b0_224_newpreprocess_decay0.94/
DATASET_DIR=/home/work/wangfei11/ImgageNetTF/
MODEL_NAME="efficientnet_b0"

CUDA_VISIBLE_DEVICES=0,1 python -u train_image_classifier.py \
    --num_clones=2 \
    --label_count=1001 \
    --train_image_count=1281167 \
    --train_dir=${TRAIN_DIR} \
    --save_summaries_secs=3600 \
    --save_interval_secs=3600 \
    --weight_decay=0.0001 \
    --learning_rate=0.02 \
    --learning_rate_decay_factor=0.94 \
    --num_epochs_per_decay=2 \
    --dataset_name="scene_singlelabel" \
    --dataset_split_name="train" \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --preprocessing_name="scene_new" \
    --train_image_size=224 \
    --batch_size=240 \
    --max_number_of_steps=1000000
    