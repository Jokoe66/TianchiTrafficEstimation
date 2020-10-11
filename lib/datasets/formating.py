import numpy as np
from mmcls.datasets.builder import PIPELINES
from mmcls.datasets.pipelines import to_tensor

@PIPELINES.register_module()
class ImagesToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            for i in range(len(results[key])):
                img = results[key][i]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                results[key][i] = to_tensor(
                    np.ascontiguousarray(img.transpose(2, 0, 1)))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class AssignImgFields(object):
    """ Add fields.
    Args:
        keys(list[str]): names of image fields that are used for transformation
                               like Crop and Resize.
        extra_aug_fields(list[str]): names of fields that are augmented to a list, used
                               during test time augmentation.

    """
    def __init__(self, keys, extra_aug_fields=None):
        self.keys = keys
        self.extra_aug_fields = extra_aug_fields

    def __call__(self, results):
        results['img_fields'] = self.keys[:] #copy
        if self.extra_aug_fields is not None:
            results['extra_aug_fields'] = self.extra_aug_fields[:] #copy
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(keys={self.keys}, extra_aug_fields={self.extra_aug_fields})'
        return repr_str
