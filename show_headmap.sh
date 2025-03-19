/c/Anaconda365/python -u show_heatmap.py \
--imagelist_path=g:/work/20200709_dxo_texture_heatmap/list.txt \
--pb_path=g:/_OnlineModel/scene_dxo_texture/20200706/scene_dxo_texture_mv1qc_4_20200706_49087_quant.pb \
--labels_file=g:/_OnlineModel/scene_dxo_texture/labels_4.txt \
--output_node_names=MobilenetV1/Predictions \
--image_size=224 \
--preprocess_type=scene_new \
--last_node_names=MobilenetV1/MobilenetV1/Conv2d_13_pointwise/Relu \
--last_node_size=1024 \
--out_dir=g:/work/20200709_dxo_texture_heatmap