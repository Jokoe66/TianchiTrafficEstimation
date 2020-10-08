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
                results[key][i] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.register_module()
class AssignImgFields(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        results['img_fields'] = self.keys
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'
