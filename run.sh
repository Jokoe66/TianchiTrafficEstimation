python cal_features.py --img_root /tcdata/amap_traffic_final_test_data \
    --ann_file /tcdata/amap_traffic_final_test_0906.json \
    --split test --device cuda:0 --save_tag final
python demo.py
