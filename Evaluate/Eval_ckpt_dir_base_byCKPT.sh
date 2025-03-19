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

echo "dir: " ${dir}
echo "preprocess_name: " ${preprocess_name}
echo "img_size: " ${img_size}
echo "LABEL_NUM: " ${LABEL_NUM}
echo "net_type: " ${net_type}
echo "output_node_names: " ${output_node_names}


while true
do
    checkpoint_steps=`ls ${dir}/model.ckpt-*.index | awk -F"-" '{print $NF}' | cut -d"." -f1 | sort -nr`
    for one_ckpt_step in ${checkpoint_steps[@]}
    do
        ckpt_path=${dir}/model.ckpt-${one_ckpt_step}
        pb_path=${dir}/model.ckpt-${one_ckpt_step}.pb
        score_file=${dir}/result_score_${one_ckpt_step}.txt
        echo "-----------------------------------------------------------------------------" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${ckpt_path}" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path}" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${score_file}" >&2
        if [ ! -e ${score_file} ]
        then
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${score_file} NOT EXISTS, Will Convert" >&2
            CUDA_VISIBLE_DEVICES=""  python export_inception_model_xiaomi.py --check_point=${ckpt_path} --graph_path=${pb_path} --labels_number=${LABEL_NUM} --net_type=${net_type} --image_size=${img_size}
            
            gpu=`nvidia-smi  | grep Default | awk '{print NR-1"\t"$9}' | sort -k2 -n | head -n1 | awk '{print $1}'`
            echo "---------------------gpu: " ${gpu}
            CUDA_VISIBLE_DEVICES="${gpu}"   python -u compute_scores_byCKPT.py \
                --net_type=${net_type} \
                --labels_number=${LABEL_NUM} \
                --img_size=${img_size} \
                --check_point=${ckpt_path} \
                --output_node_names=${output_node_names} \
                --image_gt_list=/home/work/wangfei11/TestCase/new_testimg_bycls_size_suffix/imListTest_gt_${LABEL_NUM}.txt \
                --img_root_path=/ \
                --preprocess_name=${preprocess_name} \
                --result_file=${score_file} 
                
            python -u getThreshold_${LABEL_NUM}_wide.py ${score_file}
            
            python -u getCrossMatrix_${LABEL_NUM}.py ${score_file}
        else
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} Already EXISTS, Will Skip" >&2
        fi
    done
    
    sleep 60
done

