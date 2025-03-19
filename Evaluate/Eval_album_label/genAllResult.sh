result_file=$1
badcase_file="${result_file}.badcase"

awk -F"|" -f genBadcases.awk scene_label_map_food_bigcate.txt ${result_file} > ${badcase_file}

awk -F"|" '$2!=$3' ${badcase_file} | awk -F"|" '{print $3}' | sort | uniq | xargs -i ./genNotPrecise.sh {} ${badcase_file}

awk -F"|" '$2!=$3' ${badcase_file} | awk -F"|" '{print $2}' | sort | uniq | xargs -i ./genNotRecall.sh {} ${badcase_file}
