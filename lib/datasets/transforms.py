import inspect
import math
import random

import torch
import mmcv
import numpy as np
from mmcls.datasets.builder import PIPELINES

@PIPELINES.register_module()
class PadSeq(object):
    """Pad sequence.
    Args:
        seq_len_max (int): max length of sequence.
    """

    def __init__(self,
                 seq_len_max=5,
                 pad_value=(123.675, 116.28, 103.53),
                 keys=None):
        self.seq_len_max = seq_len_max
        self.pad_value = pad_value
        self.keys = keys

    def __call__(self, results):
        if self.keys:
            results['seq_len'] = len(results[self.keys[0]])
        for key in self.keys:
            num_pad = self.seq_len_max - len(results[key])
            pad = [np.zeros_like(results[key][0]) + self.pad_value
                   for _ in range(num_pad)]
            results[key] += pad
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'seq_len_max={self.seq_len_max}, '
                    f'pad_value={self.pad_value})')
        return repr_str


@PIPELINES.register_module()
class StackSeq(object):
    """Stack sequence.
    Args:
        axis (int): which axis to stack sequence(default: last axis).
    """

    def __init__(self, axis=-1, keys=None):
        self.axis = axis
        self.keys = keys or []

    def __call__(self, results):
        for key in self.keys:
            if isinstance(results[key][0], torch.Tensor):
                results[key] = torch.stack(results[key], self.axis)
            elif isinstance(results[key][0], np.ndarray):
                results[key] = np.stack(results[key], self.axis)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'axis={self.axis})')
        return repr_str


@PIPELINES.register_module()
class SeqRandomResizedCrop(object):
    """Crop the given sequence images to random size and aspect ratio.
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

    def __call__(self, results):
        """
        Args:
            imgs (list[ndarray]): list of images to be cropped and resized.
        Returns:
            ndarray: Randomly cropped and resized images.
        """
        for key in results.get('img_fields', ['imgs']):
            imgs = results[key]
            xmin, ymin, target_height, target_width = self.get_params(
                imgs[0], self.scale, self.ratio)
            for i in range(len(imgs)):
                img = mmcv.imcrop(
                    imgs[i],
                    np.array([
                        ymin, xmin, ymin + target_width - 1,
                        xmin + target_height - 1
                    ]))
                results[key][i] = mmcv.imresize(
                    img,
                    tuple(self.size[::-1]),
                    interpolation=self.interpolation,
                    backend=self.backend)
        return results

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(size={self.size}'
        format_string += f', scale={tuple(round(s, 4) for s in self.scale)}'
        format_string += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        format_string += f', interpolation={self.interpolation})'
        return format_string


@PIPELINES.register_module()
class SeqRandomFlip(object):
    """Flip the images randomly.
    Flip the images randomly based on flip probaility and flip direction.
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

    def __call__(self, results):
        """Call function to flip image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """
        flip = True if np.random.rand() < self.flip_prob else False
        results['flip'] = flip
        results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            for key in results.get('img_fields', ['imgs']):
                for i in range(len(results[key])):
                    results[key][i] = mmcv.imflip(
                        results[key][i], direction=results['flip_direction'])
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(flip_prob={self.flip_prob})'


@PIPELINES.register_module()
class SeqResize(object):
    """Resize images.
    Args:
        size (int | tuple): Images scales for resizing (h, w).
            When size is int, the default behavior is to resize an image
            to (size, size). When size is tuple and the second value is -1,
            the short edge of an image is resized to its first value.
            For example, when size is 224, the image is resized to 224x224.
            When size is (224, -1), the short side is resized to 224 and the
            other side is computed based on the short side, maintaining the
            aspect ratio.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".
            More details can be found in `mmcv.image.geometric`.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Default: `cv2`.
    """

    def __init__(self, size, interpolation='bilinear', backend='cv2'):
        assert isinstance(size, int) or (isinstance(size, tuple)
                                         and len(size) == 2)
        self.resize_w_short_side = False
        if isinstance(size, int):
            assert size > 0
            size = (size, size)
        else:
            assert size[0] > 0 and (size[1] > 0 or size[1] == -1)
            if size[1] == -1:
                self.resize_w_short_side = True
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.size = size
        self.interpolation = interpolation
        self.backend = backend

    def _resize_imgs(self, results):
        for key in results.get('img_fields', ['imgs']):
            for i in range(len(results[key])):
                img = results[key][i]
                ignore_resize = False
                if self.resize_w_short_side:
                    h, w = img.shape[:2]
                    short_side = self.size[0]
                    if (w <= h and w == short_side) or (h <= w
                                                        and h == short_side):
                        ignore_resize = True
                    else:
                        if w < h:
                            width = short_side
                            height = int(short_side * h / w)
                        else:
                            height = short_side
                            width = int(short_side * w / h)
                else:
                    height, width = self.size
                if not ignore_resize:
                    img = mmcv.imresize(
                        img,
                        size=(width, height),
                        interpolation=self.interpolation,
                        return_scale=False,
                        backend=self.backend)
                    results[key][i] = img
                    results['img_shape'] = img.shape

    def __call__(self, results):
        self._resize_imgs(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@PIPELINES.register_module()
class SeqNormalize(object):
    """Normalize the image.
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['imgs']):
            for i in range(len(results[key])):
                results[key][i] = mmcv.imnormalize(results[key][i], self.mean, self.std,
                                                   self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={list(self.mean)}, '
        repr_str += f'std={list(self.std)}, '
        repr_str += f'to_rgb={self.to_rgb})'
        return repr_str
