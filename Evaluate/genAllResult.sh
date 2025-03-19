sed 's#/home/work/wangfei11/data/AI_scene_back/Test_IMGS/#G:/MI_Corpus/AICameraScene/Test_IMGS/_test_imgs/scene_test/#' RESULT/*matrix_*badcase.txt | sed 's#/sdcard/_test_imgs/scene_test/#G:/MI_Corpus/AICameraScene/Test_IMGS/_test_imgs/scene_test/#' > RESULT/badcases_win_path.txt

cat RESULT/badcases_win_path.txt | awk '{print $3}' | sort | uniq | xargs -i ./genNotPrecise.sh {}

cat RESULT/badcases_win_path.txt | awk '{print $2}' | sort | uniq | awk -F"#" '{for(i=1; i<=NF; i++){print $i}}' | sort | uniq | xargs -i ./genNotRecall.sh {}
