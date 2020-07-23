##下载数据集命令行
# %cd /content/drive/My Drive/TianchiTrafficEstimation-master
# !wget -P /content/drive/My\ Drive/TianchiTrafficEstimation-master/data https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531809/round1/amap_traffic_train_0712.zip
# !wget -P /content/drive/My\ Drive/TianchiTrafficEstimation-master/data https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531809/round1/amap_traffic_annotations_train.json
# %cd data
# !unzip amap_traffic_train_0712.zip && mv amap_traffic_train_0712 amap_traffic_train

import argparse
from PIL import Image
import pickle  # idx =20, 64, 77
import numpy as np
import os
from mmdet.apis import inference
from lib import ImageSequenceDataset
import numpy as np
from lib.lanedet.utils.config import Config
from lib.lanedet.inference import init_model, inference_model, show_result
from lib.utils.visualize import show_lanes
from lib.utils.geometry import split_rectangle, point_in_polygon

##车辆检测初始化
config = ('/content/drive/My Drive/TianchiTrafficEstimation-master/lib/mmdetection/configs/'
          'cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py')
checkpoint = ('https://open-mmlab.s3.ap-northeast-2.amazonaws.com/mmdetection'
              '/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco/'
              'cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-'
              '0.365_20200504_174711-4af8e66e.pth')

detector = inference.init_detector(config, checkpoint=checkpoint, device='cuda:0')
# 道路检测初始化
config_file = 'lib/lanedet/configs/culane.py'  # /path/to/config
config = Config.fromfile(config_file)
config.test_model = '/content/drive/My Drive/culane_18.pth'  # /path/to/model_weight
model = init_model(config, 'cuda:0')
##训练集初始化
parser = argparse.ArgumentParser()
parser.add_argument('--key_frame_only', action='store_true',
                    default=False)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--max_epoch', type=int, default=12)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--milestones', nargs='+', type=int,
                    default=[8, ])

args = parser.parse_args(args=[])
training_set = ImageSequenceDataset('train',
                                    key_frame_only=args.key_frame_only)
id_dis_status_variance = []#输出数据集
# {id:   status:   distance:   variance}
for idx in range(1200, training_set.__len__()):
    dis_dic = {}  # id_dis_status_variance中的一项
    dic = training_set.__getitem__(idx)
    #img_tensor = dic.get("imgs").permute(3, 0, 1, 2)
    piece = training_set.anns[idx]
    dis_dic["id"] = piece.get("id")
    print(piece.get("id"))
    dis_dic["status"] = piece.get("status")
    dis = []
    for i in range(0, dic.get("len_seq")):

        img_arry = os.path.join('data/amap_traffic_train/',
                                dis_dic.get("id"), piece.get("frames")[i].get('frame_name'))
        # img_arry = img_tensor[i].numpy()

        box_out, seg_out = inference.inference_detector(detector, img_arry)
        ##车辆检测
        vehicle_labels = ['car', 'motorcycle', 'bus', 'truck', 'bicycle', ]
        vehicle_ids = [detector.CLASSES.index(label) for label in vehicle_labels]

        box_result = [np.empty((0, 5)) for i in range(len(box_out))]
        seg_result = [[] for i in range(len(box_out))]
        for id in vehicle_ids:
            box_result[id] = box_out[id]
            seg_result[id] = seg_out[id]
        img = detector.show_result(img_arry,
                                   (box_result, seg_result),
                                   score_thr=0.5)[..., ::-1]
        img = Image.fromarray(img)
        ##路线检测    
        result = inference_model(model, img_arry)

        ##绘制
        w, h = img.size
        lines = [line[line[:, 0] > 0] for line in result if len(line[line[:, 0] > 0]) > 2]
        lanes = split_rectangle(lines, (w, h))
        main_lane = [point_in_polygon([w / 2, h], _) for _ in lanes].index(True)
        ##找最小线
        vehicles = np.vstack(box_result)
        vehicles = vehicles[vehicles[:, -1] >= 0.5]

        bottom_centers = vehicles[:, 2:4]
        bottom_centers[:, 0] -= 0.5 * (vehicles[:, 2] - vehicles[:, 0])
        if bottom_centers.size != 0:#判断有没有车
            inside_main_lane = np.array(
                [point_in_polygon(bc, lanes[main_lane]) for bc in bottom_centers])

            inside_bottom_centers = bottom_centers[inside_main_lane]
            
            if inside_bottom_centers.size == 0:  # 当车辆高于判断边界，x方向在区域中，但是y要高于多边形，
                x_point = np.sort(lanes[main_lane][:, 0], )
                y_point = np.min(lanes[main_lane][:, 1], )
                above_main_lane = np.array(
                    [bc[0] > x_point[1] and bc[0] < x_point[-2] and bc[1] < y_point for bc in bottom_centers])
                above_bottom_centers = bottom_centers[above_main_lane]
                if above_bottom_centers.size == 0:  #
                    close_car_dis = w + h
                else:
                    inside_bottom_centers = above_bottom_centers
                    closest_inside_bottom_centers = inside_bottom_centers[
                        inside_bottom_centers[:, 1] == inside_bottom_centers[:, 1].max()]
                    close_dis = []
                    for ibc in closest_inside_bottom_centers:
                        close_dis.append(h - ibc[1])

                    close_car_dis = min(close_dis)
            else:
                closest_inside_bottom_centers = inside_bottom_centers[
                    inside_bottom_centers[:, 1] == inside_bottom_centers[:, 1].max()]
                close_dis = []
                for ibc in closest_inside_bottom_centers:
                    close_dis.append(h - ibc[1])

                close_car_dis = min(close_dis)

        else:# 判断了主车道但上面没车：
            close_car_dis = w + h
        dis.append(close_car_dis /)  # close_car_dis
    dis_dic["distance"] = dis
    dis_dic["variance"] = np.std(np.array(dis))
    id_dis_status_variance.append(dis_dic)

with open('id_dis_status_variance' + '.pkl', 'wb') as f:
    pickle.dump(id_dis_status_variance, f, )
