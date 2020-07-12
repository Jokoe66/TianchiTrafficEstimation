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
  * Build 3 APIs in [inference.py](lib/lanedet/inference.py) (```init_model```, ```inference_model``` and ```show_result```)
  * Support single image test by running inference.py（See [run.sh](lib/lanedet/run.sh)).
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

