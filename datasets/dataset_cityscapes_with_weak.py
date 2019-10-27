import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import Cityscapes

from datasets.definitions import *


class DatasetCityscapesWithWeak(Dataset):
    """
    This dataset extends CityScapes dataset [1] to form a rich playground for weakly-supervised semantic image
    segmentation. It provides 2975 training and 500 validation multimodal samples, each containing the following
    modalities (other than RGB):
        - Dense semantic segmentation (overlaps are resolved in favor of VOC as those have more detail)
        - Semantic clicks (simulated them from instance segmentation as described in [2])
    Files named definitions.py and cityscapes_synthetic_clicks.json are required on the same level for this
    file to work. Please consider using the following bibtex for citation:
    [1]: @inproceedings{cordts2016cityscapes,
             title={The cityscapes dataset for semantic urban scene understanding},
             author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and
                     Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
             booktitle={CVPR},
             pages={3213--3223},
             year={2016}
         }
    [2]: @article{obukhov2019gated,
             author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
             title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
             journal={CoRR},
             volume={abs/1906.04651},
             year={2019},
             url={http://arxiv.org/abs/1906.04651},
         }
    """

    def __init__(self, dataset_root, split, download=False, integrity_check=True):
        assert not download, 'Downloading of CityScapes is not implemented in torchvision'
        assert split in (SPLIT_TRAIN, SPLIT_VALID), f'Invalid split {split}'
        self.integrity_check = integrity_check

        self.ds = Cityscapes(dataset_root, split=split, mode='fine', target_type='semantic')

        self.sample_names = []
        self.sample_id_to_name, self.sample_name_to_id = {}, {}
        for i, path in enumerate(self.ds.images):
            name = self._sample_name(path)
            self.sample_names.append(name)
            self.sample_id_to_name[i] = name
            self.sample_name_to_id[name] = i

        self.transforms = None

        dir = os.path.dirname(__file__)
        path_points = os.path.join(dir, 'cityscapes_synthetic_clicks.json')
        with open(path_points, 'r') as f:
            self.ds_clicks = json.load(f)

        self._semseg_class_colors = [clsdesc.color for clsdesc in self.ds.classes if not clsdesc.ignore_in_eval]
        self._semseg_class_names = [clsdesc.name for clsdesc in self.ds.classes if not clsdesc.ignore_in_eval]
        self._id_2_trainid = {clsdesc.id: clsdesc.train_id for clsdesc in self.ds.classes}
        self._trainid_2_id = [clsdesc.id for clsdesc in self.ds.classes if not clsdesc.ignore_in_eval]
        self._semseg_class_histogram = self._compute_histogram()

        if integrity_check:
            n_samples = len(self.sample_names)
            assert n_samples == {SPLIT_TRAIN: 2975, SPLIT_VALID: 500}[split], f'Wrong number of samples {n_samples}'
            for i, name in enumerate(self.sample_names):
                assert name in self.ds_clicks, f'Sample {i} name {name} path {self.ds.images[i]} does not have a click'
            self.integrity_check = False

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get(self, index, override_transforms=None):
        rgb, ss_dense = self.ds[index]
        width, height = rgb.size

        # relabel id to train_id in dense
        ss_dense_np = np.array(ss_dense)
        tmp = np.full((height, width), self.ignore_label, dtype=ss_dense_np.dtype)
        for srclbl, dstlbl in self._id_2_trainid.items():
            if dstlbl < 0:
                continue
            mask_src = ss_dense_np == srclbl
            tmp[mask_src] = dstlbl
        ss_dense = Image.fromarray(tmp)

        # relabel id to train_id in clicks
        name = self.sample_id_to_name[index]
        ss_clicks = self.ds_clicks[name]
        tmp = []
        for d in ss_clicks:
            clsid = d['cls']
            if self.ds.classes[clsid].ignore_in_eval:
                continue
            clsid = self._id_2_trainid[clsid]
            tmp.append((clsid, [(d['x'], d['y'])]))
        ss_clicks = tmp

        validity = Image.new("L", (width, height), color=1)

        out = {
            MOD_ID: index,
            MOD_RGB: rgb,
            MOD_SS_DENSE: ss_dense,
            MOD_SS_CLICKS: ss_clicks,
            MOD_VALIDITY: validity,
        }

        if override_transforms is not None:
            out = override_transforms(out)
        elif self.transforms is not None:
            out = self.transforms(out)

        return out

    def name_from_index(self, index):
        return self.sample_names[index]

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return len(self.sample_names)

    @staticmethod
    def _sample_name(path_rgb):
        basename = path_rgb.split('/')[-1].split('.')[0]
        tokens = basename.split('_')
        sample_name = '_'.join(tokens[:-1])
        return sample_name

    @property
    def num_classes(self):
        return len(self._semseg_class_names)

    @property
    def ignore_label(self):
        return 255

    @property
    def rgb_mean(self):
        return [255 * 0.485, 255 * 0.456, 255 * 0.406]

    @property
    def rgb_stddev(self):
        return [255 * 0.229, 255 * 0.224, 255 * 0.225]

    @property
    def semseg_class_colors(self):
        return self._semseg_class_colors

    @property
    def semseg_class_names(self):
        return self._semseg_class_names

    @property
    def semseg_class_histogram(self):
        return self._semseg_class_histogram

    def _compute_histogram(self):
        clicks_histogram = [0, ] * self.num_classes
        for name in self.sample_names:
            clicks = self.ds_clicks[name]
            for click in clicks:
                clsid = click['cls']
                if self.ds.classes[clsid].ignore_in_eval:
                    continue
                clsid = self._id_2_trainid[clsid]
                clicks_histogram[clsid] += 1
        return clicks_histogram

if __name__ == '__main__':
    import sys
    print('Checking dataset integrity...')
    cityscapesweak_train = DatasetCityscapesWithWeak(sys.argv[1], SPLIT_TRAIN, download=False, integrity_check=True)
    cityscapesweak_valid = DatasetCityscapesWithWeak(sys.argv[1], SPLIT_VALID, download=False, integrity_check=True)
    print('Dataset integrity check passed')
