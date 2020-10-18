python cal_features.py --img_root /tcdata/amap_traffic_final_b_test_data \
    --ann_file /tcdata/amap_traffic_final_b_test_1009.json \
    --split test --device cuda:0 --save_tag final

#python e2e_demo.py --img_root /tcdata/amap_traffic_final_test_data \
#    --ann_file ../user_data/enriched_annotations_test_final.pkl \
#    --test_file /tcdata/amap_traffic_final_test_0906.json \
#    --model_path ../user_data/res50/\*.pth \
#    --device cuda:0 --ensemble 5 \
#    --config_file configs/classifiers/classifier_r50_cls_or_aug.py

#python gen_dnn_feats.py --img_root /tcdata/amap_traffic_final_test_data \
#    --ann_file ../user_data/enriched_annotations_test_final.pkl \
#    --device cuda:0 \
#    --config_file configs/classifiers/classifier_r50_cls_or_aug.py

python -u lib/mseg-semantic/mseg_semantic/tool/universal_demo.py \
    --config=lib/mseg-semantic/mseg_semantic/config/test/default_config_360_ss.yaml \
    model_name mseg-3m model_path ../user_data/mseg-3m.pth \
    input_file /tcdata/amap_traffic_final_b_test_data/

python gen_seg_feats.py --ann_file ../user_data/enriched_annotations_test_final.pkl \
    --feat_file lib/mseg-semantic/temp_files/\
    mseg-3m_amap_traffic_final_b_test_data_universal_ss/360/gray/label_maps.pkl

python demo.py
