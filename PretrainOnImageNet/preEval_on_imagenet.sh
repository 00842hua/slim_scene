
if [ $# -lt 3 ]
then
    echo "Need CKPT_PATH MODEL_NAME gpu"
    exit
fi

CKPT_PATH=$1
MODEL_NAME=$2
gpu=$3

preprocessing_name="scene_new"
if [ $# -ge 4 ]
then
    preprocessing_name=$4
fi

echo "-------- $0 CKPT_PATH:          ${CKPT_PATH}"
echo "-------- $0 MODEL_NAME:         ${MODEL_NAME}"
echo "-------- $0 gpu:                ${gpu}"
echo "-------- $0 preprocessing_name: ${preprocessing_name}"

#MODEL_NAME="efficientnet_b0"
DATASET_DIR=/home/work/wangfei11/ImgageNetTF/


CUDA_VISIBLE_DEVICES=${gpu} python -u ../eval_image_classifier.py \
    --label_count=1001 \
    --checkpoint_path=${CKPT_PATH} \
    --eval_dir=${CKPT_PATH} \
    --dataset_name="scene_singlelabel" \
    --dataset_split_name="validation" \
    --dataset_dir=${DATASET_DIR} \
    --model_name=${MODEL_NAME} \
    --preprocessing_name="scene_new" \
    --eval_image_size=224 \
    --batch_size=100