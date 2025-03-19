sed 's#/home/work/wangfei11/data/AI_scene_back/Test_IMGS/#G:/MI_Corpus/AICameraScene/Test_IMGS/_test_imgs/scene_test/#' RESULT/*matrix_*badcase.txt | sed 's#/home/work/wangfei11/data/video_labeling/Test_IMGS_SmallCate/#G:/MI_Corpus/AICameraScene/Test_IMGS/_test_imgs/video_test_imgs/#' |sed 's#/home/work/wangfei11/data/AI_scene_central/Test_IMGS/#G:/MI_Corpus/AI_scene_central/Test_IMGS/scene_test_central/#' |sed 's#/home/work/wangfei11/data/AI_scene_dxo_texture/Test_IMGS/#G:/MI_Corpus/AI_scene_dxo_texture/20200703_dxo_texture/Test_IMGS/#' |sed 's#/home/work/wangfei11/data/AI_scene_dxo_2020/Test_IMGS/#G:/MI_Corpus/AI_scene_dxo_2020/Test_IMGS/#' | sed 's/%23/#/g' > RESULT/badcases_win_path.txt

cat RESULT/badcases_win_path.txt | awk '{print $3}' | sort | uniq | xargs -i ./genNotPrecise.sh {}

cat RESULT/badcases_win_path.txt | awk '{print $2}' | sort | uniq | awk -F"#" '{for(i=1; i<=NF; i++){print $i}}' | sort | uniq | xargs -i ./genNotRecall.sh {}
