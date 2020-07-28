# [天池交通状况预测比赛](https://tianchi.aliyun.com/competition/entrance/531809/information)

### Result
|       Method        | F1_0 | F1_1 | F1_2 | score |
|       :---:         | :---:| :---:| :---:| :---: |
|time based baseline  | 0.905|0.034 | 0.713| 0.616 |
|+vehicle+lane        | 0.930|0.136 | 0.770| 0.675 |
|+class balance       | 0.921|0.305 | 0.777|**0.712**|

### Taks1: Scene Recognition
Directly classfy scene images into several traffic status (unimpeded, congested and slow), based on the deep convolutional features.

|       Method        | F1_0 | F1_1 | F1_2 | score |
|       :---:         | :---:| :---:| :---:| :---: |
|multi-classification | 0.84 | 0    |  0   | 0.17  |


### Task2: Visual Odometry

|                       Paper                                             |      Year      |                    Code                      |
|                       :----:                                            |      :--:      |                    :--:                      |
|[Visual Odometry Revisited: What Should Be Learnt?](https://arxiv.org/abs/1909.09803)            | ICRA2020 |[Pytorch](https://github.com/Huangying-Zhan/DF-VO)|
|[DeepVO : Towards Visual Odometry with Deep Learning ](http://senwang.gitlab.io/DeepVO/files/wang2017DeepVO.pdf) | ICRA2017 | [Pytorch](https://github.com/ChiWeiHsiao/DeepVO-pytorch)  |
|[Unsupervised Learning of Depth and Ego-Motion from Video](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf)| CVPR2017 |[Pytorch](https://github.com/ClementPinard/SfmLearner-Pytorch) [TensorFlow](https://github.com/tinghuiz/SfMLearner) |
|[Fast, Robust, Continuous Monocular Egomotion Computation](https://arxiv.org/abs/1602.04886)| ICRA2016| None |

### Task3: Road Lane detection
#### Traditional methods
[Road lane detection based on hough line detection algorithm](https://github.com/naokishibuya/car-finding-lane-lines)

#### SOTA 
Refer to [awesome-lane-detection](https://github.com/amusi/awesome-lane-detection)

|                       Paper                                             |      Year      |                    Code                      |
|                       :----:                                            |      :--:      |                    :--:                      |
|[Ultra Fast Structure-aware Deep Lane Detection](https://arxiv.org/abs/2004.11757)            | ECCV2020 |[Pytorch](https://github.com/cfzd/Ultra-Fast-Lane-Detection)|
|[Inter-Region Affinity Distillation for Road Marking Segmentation](https://arxiv.org/abs/2004.05304)| CVPR2020 | [Pytorch](https://github.com/cardwing/Codes-for-IntRA-KD)|
|[key points estimation and point instance segmentation approach for lane detection](https://arxiv.org/abs/2002.06604)| arxiv2020 | [Pytorch](https://github.com/koyeongmin/PINet)|

#### lanedet usage
lanedet is modified for easier usage from [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)。
* The demo code is refactored.
  * Build 3 APIs in [inference.py](https://github.com/Jokoe66/Ultra-Fast-Lane-Detection/blob/63cafe63b871243818521d7d0ed3e7d044496f53/inference.py) (```init_model```, ```inference_model``` and ```show_result```)
  * Support single image test by running inference.py（See [run.sh](https://github.com/Jokoe66/Ultra-Fast-Lane-Detection/blob/63cafe63b871243818521d7d0ed3e7d044496f53/run.sh)).
* The project is refactored to be a package for external calls.
  ```python
  from lanedet.utils.config import Config
  from lanedet.inference import init_model, inference_model, show_result
  from utils.geometry import split_rectangle, point_in_polygon

  config_file = /path/to/config
  config = Config.fromfile(config_file)
  config.test_model = /path/to/model_weight

  model = init_model(config, 'cuda:0')
  img_file = /path/to/image
  result = inference_model(model, img_file)
  img = show_result(img_file, result)
  img.save(/path/to/output_image)
  
  # The lane detections are used to determine which is the main lane.
  lines = [_[_[:, 0] > 0] for _ in result if len(_[_[:, 0] > 0]) > 2] # filter high quality lane detections
  lanes = split_rectangle(lines, img.size)
  w, h = img.size
  main_lane = [point_in_polygon([w/2, h], lane) for lane in lanes].index(True)
  ```
### Task4: Vehicle detection
Vehicle detection is completed within general object detection pretrained with MS COCO dataset, based on [mmdetection](https://github.com/Jokoe66/mmdetection-1).

#### mmdetection usage
Refer to mmdetection [docs](https://github.com/Jokoe66/mmdetection-1/blob/master/docs/getting_started.md).
```python
import cv2
import numpy as np

from mmdet.apis import inference

config = /path/to/config # e.g. mmdetection/configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py
checkpoint = /path/to/model/weight # Download from mmdetection model_zoo

detector = inference.init_detector(config, checkpoint=checkpoint, device='cuda:0')
img_file = /path/to/image
out = inference.inference_detector(detector, img_file)

vehicle_labels = ['car', 'motorcycle', 'bus', 'truck', ]
vehicle_ids = [detector.CLASSES.index(label) for label in vehicle_labels]

result = [np.empty((0, 5)) for i in range(len(out))]
for id in vehicle_ids:
    result[id] = out[id]

img = detector.show_result(img_file, result)
cv2.imwrite(/path/to/output_image, img)
```

## Traffic Status Estimation [![Open in colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jokoe66/TianchiTrafficEstimation/blob/master/demo.ipynb)
### Method
- **Overview**

  Combining vehicle detection and lane detection, we can make a first simple traffic status hypothesis. The hypothesis
follows the 4-step pipeline: generate lane areas (polygons), determine the main lane, filter main-lane vehicles, and predict the
traffic status as a function of the distance of the closest main-lane vehicle.

- **Generate lane areas** 

  We first generate lane areas with the detected lane markers. We regress the lane lines, then split the image with lane lines, resulting
in several lane areas represented by polygon vertexes.

- **Determine the main lane** 

  The lane areas are used to judge the main lane where the car is driving on, and to filter out vehicles that 
we care. The main lane is determined by checking which lane area the bottom-center viewpoint **(w/2, h)**
locates in.

- **Filter main-lane vehicles**

  The vehicle detection results contain vehicle bounding-boxes (and sementic segmentation maps). The vehicles 
we care are those that locate on the main lane. The filtering is done by judging if the bottom-center points of the
vehicle bounding-boxes locate in the main lane area.

- **Predict traffic status**

  Based on the key vehicles that locate on the main lane, we predict the traffic status as a heuristic function of the distance
of the closest key vehicle from the camera. The distance is measured as the y-axis L1 distance in the image plane between
the bottom-center point of the bounding-box and the bottom-center viewpoint **(w/2, h)**. The function is fomulated 
based on two parameterized thresholds **thr1** and **thr2**. If the distance is below **thr1**, then the traffic status is hypothesized
as **congested**; if the distance is between **thr1** and **thr2**, the traffic status is **slow**; otherwise the traffic status is **unimpeded**.
