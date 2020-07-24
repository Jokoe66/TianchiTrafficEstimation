import pdb
import argparse
import json
import os

import tqdm
from PIL import Image
import mmcv
import torch
import torchvision.transforms as transforms
import numpy as np
from mmdet.apis import inference

from lib import ImageSequenceDataset
from lib.lanedet.utils.config import Config
from lib.lanedet.inference import init_model, inference_model, show_result
from lib.utils.visualize import show_lanes
from lib.utils.geometry import split_rectangle, point_in_polygon

parser = argparse.ArgumentParser()
parser.add_argument('--split', type=str, default='train', help='train or test')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--debug', action="store_true", default=False,
                    help='whether save visualized images')
parser.add_argument('--debug_dir', type=str, default='outputs',
                    help='directory to save visualized images')
args = parser.parse_args()

debug = args.debug
debug_dir = args.debug_dir
if debug:
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
##车辆检测初始化
config = ('lib/mmdetection/configs/cascade_rcnn/'
          'cascade_mask_rcnn_r50_fpn_1x_coco.py')
checkpoint = ('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection'
              '/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco/'
              'cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-'
              '0.365_20200504_174711-4af8e66e.pth')

detector = inference.init_detector(config,
                                   checkpoint=checkpoint, device=args.device)
# 道路检测初始化
config_file = 'lib/lanedet/configs/culane.py'  # /path/to/config
config = Config.fromfile(config_file)
config.test_model = 'lib/lanedet/checkpoints/culane_18.pth'  # /path/to/model_weight
model = init_model(config, args.device)
# 训练集初始化
training_set = ImageSequenceDataset(
    args.split,
    transform=transforms.Compose([
        lambda x:mmcv.imresize(x, (1280, 720)),
        lambda x:torch.tensor(x)]),
    key_frame_only=False)
enriched_annotations = [] #输出数据集

for idx in tqdm.tqdm(range(len(training_set))):
    data = training_set[idx]
    ann = training_set.anns[idx]  # id_dis_status_variance中的一项
    img_seq = data['imgs'].numpy().transpose(3, 0, 1, 2)
    ann['feats'] = dict(closest_vehicle_distance=[],
                        main_lane_vehicles=[],
                        total_vehicles=[])
    for i in range(data['len_seq']):
        img = img_seq[i]
        ##车辆检测
        box_out, seg_out = inference.inference_detector(detector,
                                                        img[..., ::-1]) # RGB -> BGR
        vehicle_labels = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', ]
        vehicle_ids = [detector.CLASSES.index(label) for label in vehicle_labels]

        box_result = [np.empty((0, 5)) for i in range(len(box_out))]
        seg_result = [[] for i in range(len(box_out))]
        for id in vehicle_ids:
            box_result[id] = box_out[id]
            seg_result[id] = seg_out[id]
        
        ##路线检测
        result = inference_model(model, img)

        ##绘制
        h, w = img.shape[:2]
        lines = [line[line[:, 0] > 0]
            for line in result if len(line[line[:, 0] > 0]) > 2] # filter valid lines
        lanes = split_rectangle(lines, (w, h))
        assert len(lanes) > 0
        main_lane = [point_in_polygon([w / 2, h], _) for _ in lanes].index(True)
        if debug:
            img_to_show = img.copy()
            img_to_show = detector.show_result(img_to_show,
                (box_result, seg_result), score_thr=0.5)[..., ::-1]
            img_to_show = show_result(img_to_show, result)
            img_to_show = show_lanes(img_to_show, lanes, main_lane)
            img_to_show.save(os.path.join(debug_dir,
                f"{ann['id']}-{i + 1}.jpg"))
        ##找最小线
        vehicles = np.vstack(box_result)
        # reserve high confident ones
        vehicles = vehicles[vehicles[:, -1] >= 0.5]
        ann['feats']['total_vehicles'].append(len(vehicles))

        bottom_centers = vehicles[:, 2:4] # bottom-right points
        bottom_centers[:, 0] -= 0.5 * (vehicles[:, 2] - vehicles[:, 0])
        # append a pseudo bottom center (farthest end of the mainlane)
        # to avoid empty bottom_centers
        farthest_main_lane_points = (
            lanes[main_lane][:, 1] == lanes[main_lane][:, 1].min())
        pseudo_bc = lanes[main_lane][farthest_main_lane_points].mean(0)
        bottom_centers = np.vstack([bottom_centers, pseudo_bc])
        
        inside_main_lane = np.array(
            [point_in_polygon(bc, lanes[main_lane]) for bc in bottom_centers])
        inside_main_lane[-1] = True

        main_lane_bottom_centers = bottom_centers[inside_main_lane]
        assert len(main_lane_bottom_centers) > 0
        ann['feats']['main_lane_vehicles'].append(
            len(main_lane_bottom_centers) - 1) # exclude the pseudo one

        closest_main_lane_bottom_centers = main_lane_bottom_centers[
            main_lane_bottom_centers[:, 1] == main_lane_bottom_centers[:, 1].max()]

        ann['feats']['closest_vehicle_distance'].append(
            (h - closest_main_lane_bottom_centers[0, 1]) / (h - pseudo_bc[1]))
    enriched_annotations.append(ann)

    if idx % 100 == 0 or idx == len(training_set) - 1:
        save_path = os.path.join('data', f'enriched_annotations_{args.split}.json')
        with open(save_path, 'w') as f:
            json.dump(enriched_annotations, f)

