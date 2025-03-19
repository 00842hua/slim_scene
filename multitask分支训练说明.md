# 训练脚本
`train_image_classifier_multitask.py`中实现了相关训练代码，当前仅支持`mobilenet_v2_qc`的多分支，子分支支持1.0, 0.5和0.35的mobilenet v2 qc，对应的`model_name`分别为`mobilenet_v2_qc_multibranch`、`mobilenet_v2_qc_multibranch_050`和`mobilenet_v2_qc_multibranch_035`。
```
python -u train_image_classifier_multitask.py \
 --num_clones=1 \
 --train_dir=${CKPT_DIR} \
 --dataset_name=default \
 --tf_file_pattern=banknote_%s_*.tfrecord \
 --dataset_dir=${TF_DIR} \
 --num_readers=8 \
 --num_preprocessing_threads=8 \
 --label_count_task1=${label_count_task1} \
 --label_count_task2=${label_count_task2} \
 --train_image_count=${image_count} \
 --dataset_split_name=train \
 --rotate_image=False \
 --model_name=mobilenet_v2_qc_multibranch \
 --preprocessing_name=inception_common \
 --crop_area_min=0.8 \
 --batch_size=${BATCH_SIZE} \
 --learning_rate=0.1 \
 --save_interval_secs=1800 \
 --checkpoint_path=${PRETRAIN} \
 --checkpoint_exclude_scopes='MobilenetV2_branch2,global_step' \
 --max_number_of_steps=${MAX_STEPS} \
 --train_image_size=224 \
 --label_smoothing=0.1 \
 --ignore_missing_vars=False \
 --trainable_scopes='MobilenetV2_branch2' \
 --mixup=True \
 --quantize_delay=${QUANT_DELAY} \
 --ignore_missing_vars=True
```

# 转pb
参考如下脚本即可：
```
python Evaluate/export_inception_model_xiaomi_multitask.py \
  --check_point=${ckpt_path} \
  --graph_path=${pb_path} \
  --labels_number_task1=${LABEL_NUM_task1} \
  --labels_number_task2=${LABEL_NUM_task2} \
  --net_type=${net_type} \
  --image_size=${img_size} \
  --quantize=${QUANT_FLAG} 
```
