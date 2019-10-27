import json
import os
import zipfile
from xml.etree import ElementTree

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, SBDataset
from tqdm import tqdm

from datasets.definitions import *


class DatasetVocSbdWithWeak(Dataset):
    """
    This dataset extends Pascal VOC [1] with Semantic Boundaries Dataset [2], ScribbleSup [3], and What's The Point [4]
    to form a rich playground for weakly-supervised semantic image segmentation. It provides 10582 training and 1449
    validation multimodal samples, each containing the following modalities (other than RGB):
        - Dense semantic segmentation (overlaps are resolved in favor of VOC as those have more detail)
        - Semantic scribbles (polylines)
        - Semantic clicks ([4] do not provide background clicks, so we simulate them from [3] as described in [5])
    The training split is a union of VOC2012 training split and all of SBD images, excluding VOC2012 validation images.
    The validation split is identical to VOC2012 validation split. Files named definitions.py, voc_scribbles.zip,
    voc_whats_the_point.json, and voc_whats_the_point_bg_from_scribbles.json are required on the same level for this
    file to work. Please consider using the following bibtex for citation:
    [1]: @misc{everingham2012pascal,
             author={Everingham, M. and {Van Gool}, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
             title={The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults},
             howpublished="http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"
             year={2012}
         }
    [2]: @inproceedings{bharath2011semanticcontours,
             author={Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik},
             title={Semantic Contours from Inverse Detectors},
             booktitle={ICCV},
             year={2011},
         }
    [3]: @inproceedings{lin2016scribblesup,
             title={Scribblesup: Scribble-supervised convolutional networks for semantic segmentation},
             author={Lin, Di and Dai, Jifeng and Jia, Jiaya and He, Kaiming and Sun, Jian},
             booktitle={CVPR},
             pages={3159--3167},
             year={2016}
         }
    [4]: @inproceedings{bearman2016whatsthepoint,
             title={What’s the point: Semantic segmentation with point supervision},
             author={Bearman, Amy and Russakovsky, Olga and Ferrari, Vittorio and Fei-Fei, Li},
             booktitle={ECCV},
             pages={549--565},
             year={2016},
         }
    [5]: @article{obukhov2019gated,
             author={Anton Obukhov and Stamatios Georgoulis and Dengxin Dai and Luc {Van Gool}},
             title={Gated {CRF} Loss for Weakly Supervised Semantic Image Segmentation},
             journal={CoRR},
             volume={abs/1906.04651},
             year={2019},
             url={http://arxiv.org/abs/1906.04651},
         }
    """

    def __init__(self, dataset_root, split, download=True, integrity_check=True):
        assert split in (SPLIT_TRAIN, SPLIT_VALID), f'Invalid split {split}'
        self.integrity_check = integrity_check

        root_voc = os.path.join(dataset_root, 'VOC')
        root_sbd = os.path.join(dataset_root, 'SBD')

        self.ds_voc_valid = VOCSegmentation(root_voc, image_set=SPLIT_VALID, download=download)

        if split == SPLIT_TRAIN:
            self.ds_voc_train = VOCSegmentation(root_voc, image_set=SPLIT_TRAIN, download=False)
            self.ds_sbd_train = SBDataset(
                root_sbd,
                image_set=SPLIT_TRAIN,
                download=download and not os.path.isdir(os.path.join(root_sbd, 'img'))
            )
            self.ds_sbd_valid = SBDataset(root_sbd, image_set=SPLIT_VALID, download=False)

            self.name_to_ds_id = {
                self._sample_name(path): (self.ds_sbd_train, i) for i, path in enumerate(self.ds_sbd_train.images)
            }
            self.name_to_ds_id.update({
                self._sample_name(path): (self.ds_sbd_valid, i) for i, path in enumerate(self.ds_sbd_valid.images)
            })
            self.name_to_ds_id.update({
                self._sample_name(path): (self.ds_voc_train, i) for i, path in enumerate(self.ds_voc_train.images)
            })
            for path in self.ds_voc_valid.images:
                name = self._sample_name(path)
                self.name_to_ds_id.pop(name, None)
        else:
            self.name_to_ds_id = {
                self._sample_name(path): (self.ds_voc_valid, i) for i, path in enumerate(self.ds_voc_valid.images)
            }

        self.sample_names = list(sorted(self.name_to_ds_id.keys()))
        self.transforms = None

        dir = os.path.dirname(__file__)
        path_points_fg = os.path.join(dir, 'voc_whats_the_point.json')
        path_points_bg = os.path.join(dir, 'voc_whats_the_point_bg_from_scribbles.json')
        with open(path_points_fg, 'r') as f:
            self.ds_clicks_fg = json.load(f)
        with open(path_points_bg, 'r') as f:
            self.ds_clicks_bg = json.load(f)
        self.ds_scribbles_path = os.path.join(dir, 'voc_scribbles.zip')
        assert os.path.isfile(self.ds_scribbles_path), f'Scribbles not found at {self.ds_scribbles_path}'
        self.cls_name_to_id = {name: i for i, name in enumerate(self.semseg_class_names)}
        self._semseg_class_histogram = self._compute_histogram()

        if integrity_check:
            results = []
            for i in tqdm(range(len(self)), desc=f'Checking "{split}" split'):
                results.append(self.get(i))
            for d in results:
                if d['num_clicks_bg'] == 0:
                    print(d['name'], 'has no background clicks')
                if d['num_clicks_fg'] == 0:
                    print(d['name'], 'has no foreground clicks')
            self.integrity_check = False

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get(self, index, override_transforms=None):
        ds, idx = self.name_to_ds_id[self.name_from_index(index)]

        path_rgb = ds.images[idx]
        name = self._sample_name(path_rgb)

        rgb = Image.open(path_rgb).convert('RGB')
        width, height = rgb.size

        ss_dense_path = ds.masks[idx]
        if ss_dense_path.endswith('mat'):
            ss_dense = ds._get_segmentation_target(ss_dense_path)
        else:
            ss_dense = Image.open(ss_dense_path)
        assert not self.integrity_check or ss_dense.size == rgb.size, \
            f'RGB and SEMSEG shapes do not match in sample {name}'

        ss_clicks_fg = self.ds_clicks_fg[name]
        ss_clicks_bg = self.ds_clicks_bg[name]

        ss_clicks = ss_clicks_fg + ss_clicks_bg
        ss_clicks = [(d['cls'], [(d['x'], d['y'])]) for d in ss_clicks]

        ss_scribbles = self._parse_scribble(
            name,
            known_width=width if self.integrity_check else None,
            known_height=height if self.integrity_check else None
        )

        validity = Image.new("L", (width, height), color=1)

        out = {
            MOD_ID: index,
            MOD_RGB: rgb,
            MOD_SS_DENSE: ss_dense,
            MOD_SS_CLICKS: ss_clicks,
            MOD_SS_SCRIBBLES: ss_scribbles,
            MOD_VALIDITY: validity,
        }

        if override_transforms is not None:
            out = override_transforms(out)
        elif self.transforms is not None:
            out = self.transforms(out)

        if self.integrity_check:
            return {
                'name': name,
                'num_clicks_bg': len(ss_clicks_bg),
                'num_clicks_fg': len(ss_clicks_fg),
            }

        return out

    def name_from_index(self, index):
        return self.sample_names[index]

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return len(self.sample_names)

    def _parse_scribble(self, name, known_width=None, known_height=None):
        with zipfile.ZipFile(self.ds_scribbles_path, 'r') as f:
            data = f.read(name + '.xml')
        sample_xml = ElementTree.fromstring(data)
        assert sample_xml.tag == 'annotation', f'XML error in sample {name}'
        found_size = False
        polylines = []
        for i in range(len(sample_xml)):
            if sample_xml[i].tag == 'size':
                found_size = True
                found_width, found_height = False, False
                sample_xml_size = sample_xml[i]
                for j in range(len(sample_xml_size)):
                    if sample_xml_size[j].tag == 'width':
                        assert known_width is None or int(sample_xml_size[j].text) == known_width, \
                            f'XML error in sample {name}'
                        found_width = True
                    elif sample_xml_size[j].tag == 'height':
                        assert known_height is None or int(sample_xml_size[j].text) == known_height, \
                            f'XML error in sample {name}'
                        found_height = True
                assert found_width and found_height, f'XML error in sample {name}'
            if sample_xml[i].tag == 'polygon':
                polygon = sample_xml[i]
                polygon_class, polygon_points = None, []
                for j in range(len(polygon)):
                    polygon_entry = polygon[j]
                    if polygon_entry.tag == 'tag':
                        polygon_class = polygon_entry.text
                    elif polygon_entry.tag == 'point':
                        assert polygon_entry[0].tag == 'X' and polygon_entry[1].tag == 'Y', \
                            f'XML error in sample {name}'
                        polygon_points.append((int(polygon_entry[0].text), int(polygon_entry[1].text)))
                assert polygon_class is not None and len(polygon_points) > 0, f'XML error in sample {name}'
                polylines.append((self.cls_name_to_id[polygon_class], polygon_points))
        assert found_size and len(polylines) > 0, f'XML error in sample {name}'
        # coordinate tuples have (x,y) order
        return polylines

    @staticmethod
    def _sample_name(path):
        return path.split('/')[-1].split('.')[0]

    @property
    def num_classes(self):
        return 21

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
        return [
            (0, 0, 0),  # 'background'
            (128, 0, 0),  # 'plane'
            (0, 128, 0),  # 'bike'
            (128, 128, 0),  # 'bird'
            (0, 0, 128),  # 'boat'
            (128, 0, 128),  # 'bottle'
            (0, 128, 128),  # 'bus'
            (128, 128, 128),  # 'car'
            (64, 0, 0),  # 'cat'
            (192, 0, 0),  # 'chair'
            (64, 128, 0),  # 'cow'
            (192, 128, 0),  # 'table'
            (64, 0, 128),  # 'dog'
            (192, 0, 128),  # 'horse'
            (64, 128, 128),  # 'motorbike'
            (192, 128, 128),  # 'person'
            (0, 64, 0),  # 'plant'
            (128, 64, 0),  # 'sheep'
            (0, 192, 0),  # 'sofa'
            (128, 192, 0),  # 'train'
            (0, 64, 128),  # 'monitor'
        ]

    @property
    def semseg_class_names(self):
        return [
            'background',
            'plane',
            'bike',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'table',
            'dog',
            'horse',
            'motorbike',
            'person',
            'plant',
            'sheep',
            'sofa',
            'train',
            'monitor',
        ]

    @property
    def semseg_class_histogram(self):
        return self._semseg_class_histogram

    def _compute_histogram(self):
        clicks_histogram = [0, ] * self.num_classes
        for name in self.sample_names:
            for ds in (self.ds_clicks_fg, self.ds_clicks_bg):
                clicks = ds[name]
                for click in clicks:
                    clsid = click['cls']
                    clicks_histogram[clsid] += 1
        return clicks_histogram

if __name__ == '__main__':
    import tempfile

    tmpdir = os.path.join(tempfile.gettempdir(), 'dataset_VOC_SBD')
    print(f'Temporary directory: {tmpdir}')
    print('Checking dataset integrity...')
    voc2012weak_train = DatasetVocSbdWithWeak(tmpdir, SPLIT_TRAIN, download=True, integrity_check=True)
    voc2012weak_valid = DatasetVocSbdWithWeak(tmpdir, SPLIT_VALID, download=True, integrity_check=True)
    print('Dataset integrity check passed')
