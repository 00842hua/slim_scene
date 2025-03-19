
# ./Eval_ckpt_dir.sh /home/work/wangfei11/ImgageNetCKPT/ckpt_efficientnet_b0_224_newpreprocess_decay0.94 efficientnet_b0

if [ $# -lt 2 ]
then
    echo "Usage: $0 CkptDir model_name [preprocess_name]"
    exit 1
fi

dir=$1
model_name=$2

echo "dir: " ${dir}



while true
do
    checkpoint_steps=`ls ${dir}/model.ckpt-*.index | awk -F"-" '{print $NF}' | cut -d"." -f1 | sort -nr`
    for one_ckpt_step in ${checkpoint_steps[@]}
    do
        ckpt_path=${dir}/model.ckpt-${one_ckpt_step}
        result_file=${dir}/eval_result_${one_ckpt_step}.txt
        echo "-----------------------------------------------------------------------------" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${ckpt_path}" >&2
        echo "`date +"%Y-%m-%d %H:%M:%S"` ${result_file}" >&2
        if [ ! -e ${result_file} ]
        then
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${result_file} NOT EXISTS, Will Evaluate" >&2
            
            # rm the eval_dir, or the result will be NOT accurate
            if [ -d ${ckpt_path} ]
            then
                echo "rm ${ckpt_path} -r"
                rm ${ckpt_path} -r
            fi
            
            gpu=`nvidia-smi  | grep Default | awk '{print NR-1"\t"$9}' | sort -k2 -n | head -n1 | awk '{print $1}'`
            echo "---------------------gpu: " ${gpu}
            
            ./preEval_on_imagenet.sh ${ckpt_path} ${model_name} ${gpu} $3 2>&1 | tee ${result_file}
        else
            echo "`date +"%Y-%m-%d %H:%M:%S"` ${result_file} Already EXISTS, Will Skip" >&2
        fi
    done
    
    sleep 60
done

