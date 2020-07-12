# [天池交通状况预测比赛](https://tianchi.aliyun.com/competition/entrance/531809/information)

### Taks1: Scene Recognition
Directly classfy scene images into several traffic status (smooth, congested and slow), based on the deep convolutional features.

|       Method        | F1_0 | F1_1 | F1_2 | score |
|       :---:         | :---:| :---:| :---:| :---: |
|multi-classification | 0.84 | 0    |  0   | 0.17  |
|ordinal regression   |      |      |      |       |


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
    
    config_file = /path/to/config
    config = Config.fromfile(config_file)
    config.test_model = /path/to/model_weight
    
    model = init_model(config, 'cuda:0')
    img_file = /path/to/image
    result = inference_model(model, img_file)
    img = show_result(result, img_file)
    img.save(/path/to/output_image)
  ```
### Task4: Vehicle detection
Vehicle detection is completed within general object detection pretrained with MS COCO dataset, based on [mmdetection](https://github.com/Jokoe66/mmdetection-1).

#### mmdetection usage
Refer to mmdetection [docs](https://github.com/Jokoe66/mmdetection-1/blob/master/docs/getting_started.md).
```python
import cv2

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
