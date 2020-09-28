import pdb
import argparse
import pickle
import json
import os
import sys
from collections import defaultdict
sys.path.insert(0, 'lib/mmdetection')

from PIL import Image
import tqdm
import mmcv
import torch
import torchvision.transforms as transforms
import numpy as np
from mmdet.apis import inference

from lib.datasets import ImageSequenceDataset
from lib.lanedet.utils.config import Config
from lib.lanedet.inference import init_model, inference_model, show_result
from lib.utils.visualize import show_lanes
from lib.utils.geometry import split_rectangle, point_in_polygon
from lib.bts.depthdetect import Depthdetect

parser = argparse.ArgumentParser()
parser.add_argument('--img_root', type=str, 
		    default='data/amap_traffic_final_train_data')
parser.add_argument('--ann_file', type=str, 
		    default='data/amap_traffic_final_train_0906.json')
parser.add_argument('--split', type=str, 
		    default='train', help='train or test')
parser.add_argument('--device', type=str, default='cuda:0', help='device')
parser.add_argument('--debug', action="store_true", default=False,
                    help='whether save visualized images')
parser.add_argument('--debug_dir', type=str, default='outputs',
                    help='directory to save visualized images')
parser.add_argument('--save_tag', type=str, default='')
args = parser.parse_args()

debug = args.debug
debug_dir = args.debug_dir
if debug:
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
# 车辆检测初始化
config = 'configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'
checkpoint = '../user_data/cascade_mask_rcnn_r50_fpn_20e_coco.pth'
detector = inference.init_detector(
    config, checkpoint=checkpoint, device=args.device)
## 障碍物检测初始化
config = 'configs/barrier/cascade_rcnn_r2_101_fpn_obstacles.py'
checkpoints = [f'../user_data/cascade_rcnn_r2_101_fpn_obstacles_{i}.pth'
        for i in range(0, 6)]
obs_detectors = [inference.init_detector(
    config, checkpoint=checkpoint, device=args.device)
    for checkpoint in checkpoints]
id2obs_det = defaultdict(int) # default to 0
if args.split == 'train':
    for i in range(1, 6):
        for _ in mmcv.load(f'../user_data/obstacles_coco_val_{i}.json')['images']:
            id2obs_det[_['sequence_id']] = i
# 道路检测初始化
config_file = 'lib/lanedet/configs/culane.py'  # /path/to/config
config = Config.fromfile(config_file)
config.test_model = 'lib/lanedet/checkpoints/culane_18.pth'  # /path/to/model_weight
config.test_model = '../user_data/culane_18.pth'
model = init_model(config, args.device)
# 深度估计初始化
depth = Depthdetect(checkpoint_path='../user_data/bts.pth')
# 训练集初始化
training_set = ImageSequenceDataset(
    img_root=args.img_root,
    ann_file=args.ann_file,
    split=args.split,
    transform=transforms.Compose([
        lambda x:mmcv.imresize(x, (1280, 720)),
    ]),
    key_frame_only=False)
enriched_annotations = [] #输出数据集
for idx in tqdm.tqdm(range(len(training_set))):
    data = training_set[idx]
    ann = training_set.anns[idx]
    img_seq = data['imgs'].numpy().transpose(3, 0, 1, 2)
    ann['feats'] = dict(closest_vehicle_distance=[],
                        main_lane_vehicles=[],
                        total_vehicles=[],
                        vehicle_distances_mean=[],
                        vehicle_distances_std=[],
                        vehicle_area=[],
                        lanes=[],
                        lane_length=[],
                        lane_width=[],
                        num_obstacles=[],
                        )
    for i in range(data['len_seq']):
        ann['frames'][i]['feats'] = dict()
        img = img_seq[i]
        h, w = img.shape[:2]
        #深度估计
        dep = depth.estimate(img[...,::-1]) # BGR -> RGB
        h, w = dep.shape[:2]
        dep = mmcv.imresize(dep, (int(w/ 10), int(h/ 10)))
        ann['frames'][i]['feats']['dep'] = dep
        # 障碍物检测
        box_out = inference.inference_detector(
            obs_detectors[id2obs_det[ann['id']]], img)
        box_result = np.vstack(box_out)
        preserve = box_result[:, -1] >= 0.5
        box_result = box_result[preserve]
        ann['feats']['num_obstacles'].append(len(box_result))
        ann['frames'][i]['feats']['obstacles'] = box_result

        ##车辆检测
        box_out, seg_out = inference.inference_detector(
            detector, img)
        vehicle_labels = [
            'car', 'motorcycle', 'bus', 'truck', 'bicycle', ]
        vehicle_ids = [
            detector.CLASSES.index(label) for label in vehicle_labels]

        box_result = np.vstack([box_out[id] for id in vehicle_ids])
        if len(box_result):
            seg_result = np.stack([
                seg for id in vehicle_ids for seg in seg_out[id]], axis=0)
        else:
            seg_result = np.zeros((0, h, w))
        assert box_result.shape[1] == 5
        assert seg_result.shape[1:] == (h, w)
        assert len(box_result) == len(seg_result)

        # preserve high confident ones
        preserve = box_result[:, -1] >= 0.5
        # preserve those above the bottom boundary (0.9 * h)
        # this condition filters out the poetential detection of the car that
        # the camera is mounted on.
        preserve &= (box_result[:, 3] < 0.95 * h)
        box_result = box_result[preserve]
        seg_result = seg_result[preserve]
        # Calculate the union of vehicle areas
        vehicle_mask = seg_result.any(axis=0)
        ann['feats']['vehicle_area'].append(vehicle_mask.mean())
        ann['frames'][i]['feats']['vehicles'] = box_result
        ann['frames'][i]['feats']['vehicle_mask'] = mmcv.imresize(
            vehicle_mask.astype('uint8'), (128, 72))

        ##路线检测
        result = inference_model(model, img[...,::-1]) # BGR -> RGB

        ##绘制
        lines = [line[line[:, 0] > 0]
            for line in result if len(line[line[:, 0] > 0]) > 2] # filter valid lines
        lanes = split_rectangle(lines, (w, h), bounds=(0, 1, 0.25, 1.0))
        assert len(lanes) > 0
        main_lane = [point_in_polygon([w / 2, h], _) for _ in lanes].index(True)
        ann['frames'][i]['feats']['main_lane'] = lanes[main_lane].flatten()
        if debug:
            img_to_show = img.copy()
            img_to_show = detector.show_result(img_to_show,
                (box_out, seg_out), score_thr=0.5)[..., ::-1] #BGR -> RGB
            img_to_show = show_result(img_to_show, result)
            img_to_show = show_lanes(img_to_show, lanes, main_lane)
            img_to_show.save(os.path.join(debug_dir,
                f"{ann['id']}-{i + 1}.jpg"))
        ##找最小线
        vehicles = box_result

        bottom_centers = vehicles[:, 2:4] # bottom-right points
        bottom_centers[:, 0] -= 0.5 * (vehicles[:, 2] - vehicles[:, 0])
        ann['feats']['total_vehicles'].append(len(vehicles))
        # append a pseudo bottom center (farthest end of the mainlane)
        # to avoid empty bottom_centers
        farthest_main_lane_points = (
            lanes[main_lane][:, 1] == lanes[main_lane][:, 1].min())
        pseudo_bc = lanes[main_lane][farthest_main_lane_points].mean(0)
        bottom_centers = np.vstack([bottom_centers, pseudo_bc])
        vehicle_distances = np.sqrt(
            (bottom_centers - np.array([w / 2, h])) ** 2).sum(1) / h
        ann['feats']['vehicle_distances_mean'].append(vehicle_distances.mean())
        ann['feats']['vehicle_distances_std'].append(vehicle_distances.std())
        
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
        ann['feats']['lane_length'].append((h - pseudo_bc[1]) / h)
        ann['feats']['lane_width'].append(
            (lanes[main_lane][:, 0].max() - lanes[main_lane][:, 0].min()) / w)
        ann['feats']['lanes'].append(len(lanes))
    enriched_annotations.append(ann)

    if idx % 100 == 0 or idx == len(training_set) - 1:
        save_file = f'enriched_annotations_{args.split}'
        if args.save_tag:
            save_file += f'_{args.save_tag}'
        save_file += '.pkl'
        save_path = os.path.join('../user_data', save_file)
        with open(save_path, 'wb') as f:
            pickle.dump(enriched_annotations, f)

