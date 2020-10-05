import inspect
import math
import random

import torch
import mmcv
import numpy as np
from mmcls.datasets.builder import PIPELINES

@PIPELINES.register_module()
class mResize:
    def __init__(self, size):
        self.size = size[::-1]

    def __call__(self, img):
        return mmcv.imresize(img, self.size)


@PIPELINES.register_module()
class mNormalize:

    def __init__(self, **img_norm_cfg):
        self.img_norm_cfg = dict(
            mean=np.array(img_norm_cfg['mean']),
            std=np.array(img_norm_cfg['std']),
            to_rgb=img_norm_cfg['to_rgb']
            )

    def __call__(self, img):
        img = mmcv.imnormalize(img, **self.img_norm_cfg)
        return img


@PIPELINES.register_module()
class mImageToTensor:

    def __call__(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1))


@PIPELINES.register_module()
class mRandomFlip(object):
    """Flip the image randomly.
    Flip the image randomly based on flip probaility and flip direction.
    Args:
        flip_prob (float): probability of the image being flipped. Default: 0.5
        direction (str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, flip_prob=0.5, direction='horizontal'):
        assert 0 <= flip_prob <= 1
        assert direction in ['horizontal', 'vertical']
        self.flip_prob = flip_prob
        self.direction = direction

    def __call__(self, imgs):
        """Call function to flip image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        if flip:
            # flip image
            imgs = mmcv.imflip(imgs, direction=self.direction)
        return imgs 

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class mRandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Default: (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Default: (3. / 4., 4. / 3.).
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Default:
            'bilinear'.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear',
                 backend='cv2'):
        if isinstance(size, (tuple, list)):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received {scale}')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
        Returns:
            tuple: Params (xmin, ymin, target_height, target_width) to be
                passed to ``crop`` for a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                xmin = random.randint(0, height - target_height)
                ymin = random.randint(0, width - target_width)
                return xmin, ymin, target_height, target_width

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        xmin = (height - target_height) // 2
        ymin = (width - target_width) // 2
        return xmin, ymin, target_height, target_width

    def __call__(self, img):
        """
        Args:
            img (ndarray): Image to be cropped and resized.
        Returns:
            ndarray: Randomly cropped and resized image.
        """
        xmin, ymin, target_height, target_width = self.get_params(
            img, self.scale, self.ratio)
        img = mmcv.imcrop(
            img,
            np.array([
                ymin, xmin, ymin + target_width - 1,
                xmin + target_height - 1
            ]))
        img = mmcv.imresize(
            img,
            tuple(self.size[::-1]),
            interpolation=self.interpolation,
            backend=self.backend)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', interpolation={self.interpolation})'
        return format_string
