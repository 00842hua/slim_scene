if [ $# -lt 1 ]
then
    echo "Usage: $0 CkptDir [preprocess_name(Default:scene) img_size(Default:224) LABEL_NUM(Default:36)]"
    exit 1
fi

net_type="mobilenet_v1_qc"
output_node_names="MobilenetV1/Predictions"

./Eval_ckpt_dir_base.sh ${net_type} ${output_node_names} $@

