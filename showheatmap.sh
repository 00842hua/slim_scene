pb_path=/home/ypf/project/Convert2Dlc/model/MnasNet/mnasnet_208.pb 
image_list=/home/ypf/data/heatmap_data/image_list.txt
output_node=Mnasnet/Predictions_sigmoid
last_node=Mnasnet/Conv_1/Conv2D
labels_file=/home/ypf/project/Convert2Dlc/model/MnasNet/labels_208.txt
output_dir=/home/ypf/data/heatmap_data

python show_heatmap_by_cam.py --pb_path=${pb_path}\
    --imagelist_path=${image_list} \
    --output_node_names=${output_node} \
    --last_node_names=${last_node} \
    --labels_file=${labels_file} \
    --output_dir=${output_dir}
