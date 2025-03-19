if [ $# -lt 3 ]
then
    echo "Usage: $0 NetType OutputNodeNames CkptDir [proprocess_name img_size LABEL_NUM]"
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


while true
do
    checkpoint_steps=`ls ${dir}/model.ckpt-*.index | awk -F"-" '{print $NF}' | cut -d"." -f1 | sort -nr`
    for one_ckpt_step in ${checkpoint_steps[@]}
    do
        ckpt_path=${dir}/model.ckpt-${one_ckpt_step}
        pb_path=${dir}/model.ckpt-${one_ckpt_step}.pb
        if [ ${QUANT_FLAG} -eq 1 ]
        then
            pb_path=${dir}/model.ckpt-${one_ckpt_step}.quant.pb
        fi
        score_file=${dir}/result_score_${one_ckpt_step}.txt
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
                ${TEST_IMG_GT_FILE_PATH} ${LABEL_NUM} ${TEST_IMG_GT_FILE_DIR}/../label_${LABEL_NUM}.txt
        else
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} Already EXISTS, Will Skip" >&2
        fi
    done
    
    sleep 60
done

