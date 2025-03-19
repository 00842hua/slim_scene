if [ $# -lt 3 ]
then
    echo "Usage: $0 NetType OutputNodeNames Ckpt [proprocess_name img_size LABEL_NUM]"
    exit 1
fi

net_type=$1
output_node_names=$2
dir=$3

preprocess_name="scene"
if [ $# -ge 4 ]
then
    preprocess_name=$4
fi

img_size=224
if [ $# -ge 5 ]
then
    img_size=$5
fi


LABEL_NUM=36
if [ $# -ge 6 ]
then
    LABEL_NUM=$6
fi

TEST_IMG_GT_FILE_DIR=/home/work/wangfei11/data/AI_scene_back/Test_IMGS
TEST_IMG_GT_FILE_PATH=${TEST_IMG_GT_FILE_DIR}/imListTest_gt_${LABEL_NUM}.txt


if [[ ${dir} =~ '/' ]]
then
    dir_pwd=${dir}
else
    dir_pwd=`pwd`
fi

QUANT_FLAG=0
if [[ ${dir_pwd} =~ "quant" ]]
then
    QUANT_FLAG=1
fi

echo "dir: " ${dir}
echo "dir_pwd: " ${dir_pwd}
echo "preprocess_name: " ${preprocess_name}
echo "img_size:   " ${img_size}
echo "LABEL_NUM:  " ${LABEL_NUM}
echo "net_type:   " ${net_type}
echo "QUANT_FLAG: " ${QUANT_FLAG}
echo "output_node_names: " ${output_node_names}


one_ckpt_step=`echo ${dir} | awk -F"-" '{print $NF}'`
echo "one_ckpt_step: " ${one_ckpt_step}

ckpt_path=${dir}
pb_path=${dir}.pb
if [ ${QUANT_FLAG} -eq 1 ]
then
	pb_path=${dir}.quant.pb
fi
dir_name=`dirname ${dir}`
score_file=${dir_name}/result_score_${one_ckpt_step}.txt

echo "ckpt_path: " ${ckpt_path}
echo "pb_path: " ${pb_path}
echo "dir_name: " ${dir_name}
echo "score_file: " ${score_file}

echo "-----------------------------------------------------------------------------" >&2
echo "`date +"%Y-%m-%d %H:%M:%S"` ${ckpt_path}" >&2
echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path}" >&2
echo "`date +"%Y-%m-%d %H:%M:%S"` ${score_file}" >&2
if [ ! -e ${pb_path} ]
then
	echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} NOT EXISTS, Will Convert" >&2
	CUDA_VISIBLE_DEVICES=""  python /home/work/wangfei11/slim_scene/Evaluate/export_inception_model_xiaomi.py \
		--check_point=${ckpt_path} --graph_path=${pb_path} --labels_number=${LABEL_NUM} --net_type=${net_type} \
		--image_size=${img_size} --output_node_names=${output_node_names} --quantize=${QUANT_FLAG}
	
	gpu=`nvidia-smi  | grep Default | awk '{print NR-1"\t"$11-$9-5000}' | grep -v '-' | sort -k2 -n | head -n1 | awk '{print $1}'`
	while [ ! ${gpu} ]
	do
		echo "no available gpu, will sleep 60s"
		sleep 60
		gpu=`nvidia-smi  | grep Default | awk '{print NR-1"\t"$11-$9-5000}' | grep -v '-' | sort -k2 -n | head -n1 | awk '{print $1}'`
	done
	
	echo "---------------------gpu: " ${gpu}
	CUDA_VISIBLE_DEVICES="${gpu}"   python -u /home/work/wangfei11/slim_scene/Evaluate/compute_scores.py \
		--pb_path=${pb_path} \
		--output_node_names=${output_node_names} \
		--image_gt_list=${TEST_IMG_GT_FILE_PATH} \
		--img_root_path=/ \
		--preprocess_name=${preprocess_name} \
		--result_file=${score_file} \
		--img_size=${img_size}
		
	python -u /home/work/wangfei11/slim_scene/Evaluate/getThreshold_wide_mul_thread.py ${score_file} \
		${TEST_IMG_GT_FILE_PATH}  ${LABEL_NUM} ${TEST_IMG_GT_FILE_DIR}/../label_${LABEL_NUM}.txt
	
	python -u /home/work/wangfei11/slim_scene/Evaluate/getCrossMatrix.py ${score_file} \
		${TEST_IMG_GT_FILE_PATH} ${LABEL_NUM} ${TEST_IMG_GT_FILE_DIR}/../label_${LABEL_NUM}.txt \
		/home/work/wangfei11/data/AI_scene_back/Test_IMGS/Eval_label_map_restrict.txt
else
	echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} Already EXISTS, Will Skip" >&2
fi


