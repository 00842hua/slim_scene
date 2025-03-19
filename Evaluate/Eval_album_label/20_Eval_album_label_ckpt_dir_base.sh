if [ $# -lt 5 ]
then
    echo "Usage: $0 CkptDir NetType proprocess_name img_size LABEL_NUM label_file [OutputNodeNames]"
    exit 1
fi

dir=$1
net_type=$2
preprocess_name=$3
img_size=$4
LABEL_NUM=$5
label_file=$6
output_node_names=""

label_mapping_file=/home/work/wangfei11/data/album_labeling/Test_IMGS/scene_label_name_mapping_multi_food_bigcate.txt

if [ $# -ge 7 ]
then
    output_node_names=$7
fi


echo "dir: " ${dir}
echo "net_type: " ${net_type}
echo "preprocess_name: " ${preprocess_name}
echo "img_size: " ${img_size}
echo "LABEL_NUM: " ${LABEL_NUM}
echo "output_node_names: " ${output_node_names}


export PYTHONPATH=${PYTHONPATH}:../../
while true
do
    checkpoint_steps=`ls ${dir}/model.ckpt-*.index | awk -F"-" '{print $NF}' | cut -d"." -f1 | sort -nr`
    for one_ckpt_step in ${checkpoint_steps[@]}
    do
        ckpt_path=${dir}/model.ckpt-${one_ckpt_step}
        pb_path=${dir}/model.ckpt-${one_ckpt_step}.pb
        result_img_prediction_file=${dir}/result_img_prediction_${one_ckpt_step}.txt
        result_img_prediction_file_brief_000=${dir}/result_img_prediction_${one_ckpt_step}_brief_000.txt
        result_img_prediction_file_brief_111=${dir}/result_img_prediction_${one_ckpt_step}_brief_111.txt
        result_img_prediction_file_brief_ALL=${dir}/result_img_prediction_${one_ckpt_step}_brief_ALL.txt
        mapped_result_file=${dir}/mapped_result_img_prediction_mapped_${one_ckpt_step}.txt
        acc_result_file_000=${dir}/acc_result_${one_ckpt_step}_000.txt
        acc_result_file_111=${dir}/acc_result_${one_ckpt_step}_111.txt
        acc_result_file_ALL=${dir}/acc_result_${one_ckpt_step}_ALL.txt
        echo "-----------------------------------------------------------------------------" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${ckpt_path}" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path}" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${result_img_prediction_file}" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${mapped_result_file}" >&2
        if [ ! -e ${pb_path} ]
        then
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} NOT EXISTS, Will Convert" >&2
            CUDA_VISIBLE_DEVICES=""  python /home/work/wangfei11/slim_scene/Evaluate/export_inception_model_xiaomi.py \
                --check_point=${ckpt_path} --graph_path=${pb_path} --labels_number=${LABEL_NUM} --net_type=${net_type} --image_size=${img_size}
            
            gpu=`nvidia-smi  | grep Default | awk '{print NR-1"\t"$9}' | sort -k2 -n | head -n1 | awk '{print $1}'`
            echo "---------------------gpu: " ${gpu}
            CUDA_VISIBLE_DEVICES="${gpu}" python /home/work/wangfei11/slim_scene/predict_batch.py \
                --net_type=${net_type} \
                --preprocess_type=${preprocess_name} \
                --labels_number=${LABEL_NUM} \
                --image_size=${img_size} \
                --check_point=${ckpt_path} \
                --image_list=/home/work/wangfei11/data/album_labeling/Test_IMGS/list.txt \
                --labels_file=${label_file} \
                --central_fraction=0 \
                --batch_size=50 \
                --output_node_names=${output_node_names} \
                --multi_label_output=True
                
            mv result_img_prediction.txt ${result_img_prediction_file}
            
            awk '{n=split($1, a, "/"); if (a[9]=="000") {print $1"|"a[10]"|"$3"|"$2}}'  ${result_img_prediction_file} > ${result_img_prediction_file_brief_000}
            awk '{n=split($1, a, "/"); if (a[9]=="000" || a[9]=="111") {print $1"|"a[10]"|"$3"|"$2}}'  ${result_img_prediction_file} > ${result_img_prediction_file_brief_111}
            awk '{n=split($1, a, "/"); {print $1"|"a[10]"|"$3"|"$2}}'  ${result_img_prediction_file} > ${result_img_prediction_file_brief_ALL}
            
            #awk -F"|" -f /home/work/wangfei11/data/album_labeling/Test_IMGS/calc_recall_precision.awk ${label_mapping_file} ${result_img_prediction_file_brief_000} >  ${acc_result_file_000}
            #awk -F"|" -f /home/work/wangfei11/data/album_labeling/Test_IMGS/calc_recall_precision.awk ${label_mapping_file} ${result_img_prediction_file_brief_111} >  ${acc_result_file_111}
            #awk -F"|" -f /home/work/wangfei11/data/album_labeling/Test_IMGS/calc_recall_precision.awk ${label_mapping_file} ${result_img_prediction_file_brief_ALL} >  ${acc_result_file_ALL}
            python /home/work/wangfei11/data/album_labeling/Test_IMGS/calc_recall_precision.py ${result_img_prediction_file_brief_000} ${label_mapping_file} 0 >  ${acc_result_file_000}
            python /home/work/wangfei11/data/album_labeling/Test_IMGS/calc_recall_precision.py ${result_img_prediction_file_brief_111} ${label_mapping_file} 0 >  ${acc_result_file_111}
            python /home/work/wangfei11/data/album_labeling/Test_IMGS/calc_recall_precision.py ${result_img_prediction_file_brief_ALL} ${label_mapping_file} 0 >  ${acc_result_file_ALL}

        else
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${pb_path} Already EXISTS, Will Skip" >&2
        fi
    done
    
    sleep 60
done

:'
-------- 评测 mobilenet_v2:
./20_Eval_album_label_ckpt_dir_base.sh \
/home/work/wangfei11/data/album_labeling/CKPT/CKPT_MobileNetV2_158_20190712_lr0.01_bs200 \
mobilenet_v2 \
inception \
224 \
158 \
/home/work/wangfei11/data/album_labeling/TF_158_20190712/labels.txt



'
