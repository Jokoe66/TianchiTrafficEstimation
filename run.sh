python cal_features.py --img_root /tcdata/amap_traffic_final_test_data \
    --ann_file /tcdata/amap_traffic_final_test_0906.json \
    --split test --device cuda:0 --save_tag final

python e2e_demo.py --img_root /tcdata/amap_traffic_final_test_data \
    --ann_file ../user_data/enriched_annotations_test_final.pkl \
    --test_file /tcdata/amap_traffic_final_test_0906.json \
    --model_path ../user_data/classifier_epoch1.pth \
    --device cuda:0

#python demo.py
