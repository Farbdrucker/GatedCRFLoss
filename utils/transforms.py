import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import Compose
from datasets.definitions import *


class Identity:
    def __call__(self, sample):
        return sample


class CropForPassableSidesPIL:
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, sample):
        assert MOD_RGB in sample, f'{__class__} needs to know canvas size'
        width, height = sample[MOD_RGB].size
        width_new = self.multiple * (width // self.multiple)
        height_new = self.multiple * (height // self.multiple)
        assert width_new > 0 and height_new > 0, 'Bad input dimensions'
        for modality in sample.keys():
            data = sample[modality]
            if isinstance(data, Image.Image):
                data = data.crop((0, 0, width_new, height_new))
                sample[modality] = data
        return sample


class _RandomScaledTiltedWarpedPIL:
    def __init__(
            self, dst_size, random_scale_min, random_scale_max,
            random_tilt_max_deg, random_wiggle_max_ratio, random_reflect,
            semseg_ignore_class, center_offset_instead_of_random
    ):
        assert isinstance(dst_size, int) or isinstance(dst_size, tuple), f'Invalid type of dst_size {type(dst_size)}'
        self.dst_size = dst_size if isinstance(dst_size, tuple) else (dst_size, dst_size)
        self.random_scale_min = random_scale_min
        self.random_scale_max = random_scale_max
        self.random_tilt_max_deg = random_tilt_max_deg
        self.random_wiggle_max_ratio = random_wiggle_max_ratio
        self.random_reflect = random_reflect
        self.semseg_ignore_class = semseg_ignore_class
        self.center_offset_instead_of_random = center_offset_instead_of_random

    def __call__(self, sample):
        assert MOD_RGB in sample, f'{__class__} needs to know canvas size'
        src_size = sample[MOD_RGB].size
        dst_corners = [
            np.array([0, 0], dtype=np.float32),
            np.array([0, self.dst_size[1]], dtype=np.float32),
            np.array([self.dst_size[0], self.dst_size[1]], dtype=np.float32),
            np.array([self.dst_size[0], 0], dtype=np.float32)
        ]
        if self.random_reflect:
            do_reflect = np.random.random() < 0.5
            if do_reflect:
                dst_corners = list(reversed(dst_corners))
        src_corners, src_scale = _RandomScaledTiltedWarpedPIL._generate_corners(
            src_size, self.dst_size, self.random_scale_min, self.random_scale_max,
            self.random_tilt_max_deg, self.random_wiggle_max_ratio,
            self.center_offset_instead_of_random,
        )
        warp_coef_inv = _RandomScaledTiltedWarpedPIL._perspective_transform_from_corners(dst_corners, src_corners)
        warp_coef_fwd = _RandomScaledTiltedWarpedPIL._perspective_transform_from_corners(src_corners, dst_corners)
        warp_coef_fwd = np.append(warp_coef_fwd, 1).reshape((3, 3))

        for modality in sample.keys():
            interp = MODE_INTERP[modality]
            data = sample[modality]
            if interp is None:
                continue
            elif interp in ('nearest', 'bilinear'):
                assert isinstance(data, Image.Image), f'Input must be PIL.Image, found {type(data)}'
                interp_pil = {
                    'nearest': Image.NEAREST,
                    'bilinear': Image.BILINEAR,
                }[interp]
                fill_color = {
                    MOD_RGB: None,
                    MOD_VALIDITY: 0,
                    MOD_SS_DENSE: self.semseg_ignore_class,
                }[modality]
                data = data.transform(
                    self.dst_size, Image.PERSPECTIVE, warp_coef_inv, interp_pil, fillcolor=fill_color
                )
            elif interp == 'sparse':
                new_data = []
                for item in data:
                    clsid, joints = item
                    new_joints = []
                    for pt in joints:
                        pt = np.array([pt[0], pt[1], 1.0], np.float32)
                        pt_new = np.matmul(warp_coef_fwd, pt)
                        pt_x_new = pt_new[0] / pt_new[2]
                        pt_y_new = pt_new[1] / pt_new[2]
                        new_pt = (pt_x_new, pt_y_new)
                        new_joints.append(new_pt)
                    new_item = (clsid, new_joints)
                    new_data.append(new_item)
                data = new_data
            sample[modality] = data

        return sample

    # adopted from https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    # arguments must be lists of tuples like [(x_tl, y_tl), (x_bl, y_bl), (x_br, y_br), (x_tr, y_tr)] with frame of
    # reference in the top-left corner of the image
    @staticmethod
    def _perspective_transform_from_corners(corners_src, corners_dst):
        matrix = []
        for p_src, p_dst in zip(corners_src, corners_dst):
            matrix.append([p_src[0], p_src[1], 1, 0, 0, 0, -p_dst[0] * p_src[0], -p_dst[0] * p_src[1]])
            matrix.append([0, 0, 0, p_src[0], p_src[1], 1, -p_dst[1] * p_src[0], -p_dst[1] * p_src[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(corners_dst).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res, dtype=np.float32).reshape(8)

    @staticmethod
    def _dst_corners_bounding_box(corners):
        x_min, x_max = corners[0][0], corners[0][0]
        y_min, y_max = corners[0][1], corners[0][1]
        for corner in corners[1:]:
            x_min = min(x_min, corner[0])
            x_max = max(x_max, corner[0])
            y_min = min(y_min, corner[1])
            y_max = max(y_max, corner[1])
        return x_min, x_max, y_min, y_max

    @staticmethod
    def _transform_scale_rotate_wiggle(dst_size, scale_min, scale_max, angle_max_deg, wiggle_max_ratio):
        corners = [
            np.array([-dst_size[0] / 2, -dst_size[1] / 2], dtype=np.float32),
            np.array([-dst_size[0] / 2, dst_size[1] / 2], dtype=np.float32),
            np.array([dst_size[0] / 2, dst_size[1] / 2], dtype=np.float32),
            np.array([dst_size[0] / 2, -dst_size[1] / 2], dtype=np.float32)
        ]

        max_wiggle_pix = wiggle_max_ratio * min(dst_size[0], dst_size[1]) / 2
        scale = np.random.uniform(scale_min, scale_max)
        angle_deg = np.random.uniform(-angle_max_deg, angle_max_deg) if 0 < angle_max_deg <= 45 else 0
        wiggle_factor = [
            np.array([
                np.random.uniform(-max_wiggle_pix, max_wiggle_pix),
                np.random.uniform(-max_wiggle_pix, max_wiggle_pix)
            ], dtype=np.float32) for _ in range(4)
        ]

        angle_rad = np.deg2rad(angle_deg)
        matrix_rot = np.array([
            [np.cos(angle_rad), np.sin(-angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=np.float32)

        corners = [np.matmul(matrix_rot, scale * (c + w)) for c, w in zip(corners, wiggle_factor)]
        return corners, scale

    @staticmethod
    def _generate_corners(
            src_size, dst_size, random_scale_min=1.0, random_scale_max=2.0,
            random_tilt_max_deg=0.0, random_wiggle_max_ratio=0.0, center_offset_instead_of_random=False
    ):
        assert random_scale_min > 0, 'random_scale_min must be positive'
        assert random_scale_max >= random_scale_min, 'random_scale_max > random_scale_min'
        assert random_tilt_max_deg >= 0, 'tilt must be non negative'
        assert 0 <= random_wiggle_max_ratio < 0.5, 'random_wiggle_max_ratio must be [0, 1/2]'

        corners, scale = _RandomScaledTiltedWarpedPIL._transform_scale_rotate_wiggle(
            dst_size, random_scale_min, random_scale_max, random_tilt_max_deg, random_wiggle_max_ratio
        )
        x_min, x_max, y_min, y_max = _RandomScaledTiltedWarpedPIL._dst_corners_bounding_box(corners)

        range_x_min = -x_min
        range_x_max = src_size[0] - x_max
        range_y_min = -y_min
        range_y_max = src_size[1] - y_max

        if center_offset_instead_of_random or range_x_max <= range_x_min:
            offs_x = (range_x_min + range_x_max) * 0.5
        else:
            offs_x = np.random.uniform(range_x_min, range_x_max)

        if center_offset_instead_of_random or range_y_max <= range_y_min:
            offs_y = (range_y_min + range_y_max) * 0.5
        else:
            offs_y = np.random.uniform(range_y_min, range_y_max)

        corners = [c + np.array([offs_x, offs_y], dtype=np.float32) for c in corners]
        return corners, scale


class RandomScaledTiltedWarpedCropPIL(_RandomScaledTiltedWarpedPIL):
    def __init__(
            self, dst_size, random_scale_min, random_scale_max, random_tilt_max_deg, random_wiggle_max_ratio,
            random_reflect, semseg_ignore_class
    ):
        super(RandomScaledTiltedWarpedCropPIL, self).__init__(
            dst_size, random_scale_min, random_scale_max, random_tilt_max_deg,
            random_wiggle_max_ratio, random_reflect, semseg_ignore_class,
            center_offset_instead_of_random=False
        )


class CropCenterPIL(_RandomScaledTiltedWarpedPIL):
    def __init__(self, dst_size, semseg_ignore_class):
        super(CropCenterPIL, self).__init__(
            dst_size,
            random_scale_min=1.0,
            random_scale_max=1.0,
            random_tilt_max_deg=0.0,
            random_wiggle_max_ratio=0.0,
            random_reflect=False,
            semseg_ignore_class=semseg_ignore_class,
            center_offset_instead_of_random=True,
        )


class ScribblesShrink:
    def __init__(self, ratio):
        assert 0 < ratio <= 1, f'Invalid shrinkage ratio: {ratio}'
        self.ratio = ratio

    def __call__(self, sample):
        if MOD_SS_SCRIBBLES not in sample or self.ratio == 1:
            return sample

        polylines = sample[MOD_SS_SCRIBBLES]
        new_polylines = []
        for scribble_cls, scribble_xy in polylines:
            # estimate length of the scribble
            scribble_lens = [0.0]
            for i in range(1, len(scribble_xy)):
                scribble_lens.append(scribble_lens[-1] + math.sqrt(
                    (scribble_xy[i][0] - scribble_xy[i-1][0]) ** 2 +
                    (scribble_xy[i][1] - scribble_xy[i-1][1]) ** 2
                ))
            len_half = scribble_lens[-1] * 0.5
            # find cut points
            len_low = len_half * (1.0 - self.ratio)
            len_high = len_half * (1.0 + self.ratio)
            new_scribble_xy = []
            # create a shortened scribble polyline
            for i in range(0, len(scribble_lens)-1):
                if scribble_lens[i+1] < len_low:
                    continue
                elif scribble_lens[i] < len_low <= scribble_lens[i+1]:
                    frac = 1.0 * (len_low - scribble_lens[i]) / (scribble_lens[i+1] - scribble_lens[i])
                    new_scribble_xy.append((
                        scribble_xy[i][0] + (scribble_xy[i+1][0] - scribble_xy[i][0]) * frac,
                        scribble_xy[i][1] + (scribble_xy[i+1][1] - scribble_xy[i][1]) * frac,
                    ))
                elif scribble_lens[i+1] <= len_high:
                    new_scribble_xy.append(scribble_xy[i])
                elif scribble_lens[i] <= len_high < scribble_lens[i+1]:
                    frac = 1.0 * (len_high - scribble_lens[i]) / (scribble_lens[i+1] - scribble_lens[i])
                    new_scribble_xy.append((
                        scribble_xy[i][0] + (scribble_xy[i+1][0] - scribble_xy[i][0]) * frac,
                        scribble_xy[i][1] + (scribble_xy[i+1][1] - scribble_xy[i][1]) * frac,
                    ))
                elif len_high < scribble_lens[i]:
                    continue
            new_polylines.append((scribble_cls, new_scribble_xy))

        sample[MOD_SS_SCRIBBLES] = new_polylines
        return sample


class WeakRasterizerPIL:
    def __init__(self, semseg_ignore_class, stroke_width):
        self.semseg_ignore_class = semseg_ignore_class
        self.stroke_width = stroke_width

    def __call__(self, sample):
        if MOD_SS_SCRIBBLES not in sample and MOD_SS_CLICKS not in sample:
            return sample
        assert MOD_RGB in sample, f'{__class__} needs to know canvas size'
        width, height = sample[MOD_RGB].size
        if MOD_SS_SCRIBBLES in sample:
            sample[MOD_SS_SCRIBBLES] = self.rasterize_scribbles(sample[MOD_SS_SCRIBBLES], width, height)
        if MOD_SS_CLICKS in sample:
            sample[MOD_SS_CLICKS] = self.rasterize_scribbles(sample[MOD_SS_CLICKS], width, height)
        return sample

    def rasterize_scribbles(self, data, width, height):
        img = Image.new("L", (width, height), color=self.semseg_ignore_class)
        draw = ImageDraw.Draw(img)
        polylines = data
        for clsid, joints in polylines:
            if len(joints) > 1:
                draw.line(joints, clsid, self.stroke_width, joint="curve")
            for i in range(len(joints)):
                draw.ellipse((
                    joints[i][0] - self.stroke_width / 2, joints[i][1] - self.stroke_width / 2,
                    joints[i][0] + self.stroke_width / 2, joints[i][1] + self.stroke_width / 2),
                    clsid
                )
        return img

    def rasterize_clicks(self, data, width, height):
        img = Image.new("L", (width, height), color=self.semseg_ignore_class)
        draw = ImageDraw.Draw(img)
        for clsid, click in data:
            if self.stroke_width > 1:
                draw.ellipse((
                    click[0] - self.stroke_width / 2, click[1] - self.stroke_width / 2,
                    click[0] + self.stroke_width / 2, click[1] + self.stroke_width / 2),
                    clsid
                )
            else:
                draw.point(click, clsid)
        return img


class ConvertToTensorsSimple:
    def __call__(self, sample):
        for modality in sample.keys():
            data = sample[modality]
            if modality == MOD_ID:
                data = torch.tensor(data, dtype=torch.long)
            elif modality == MOD_RGB:
                data = torch.from_numpy(np.array(data)).float().permute(2, 0, 1)
            elif modality == MOD_VALIDITY:
                data = torch.from_numpy(np.array(data)).float().unsqueeze(0)
            elif modality in (MOD_SS_DENSE, MOD_SS_SCRIBBLES, MOD_SS_CLICKS):
                data = torch.from_numpy(np.array(data)).long().unsqueeze(0)
            else:
                print(f'Unaccounted data, collate WILL complain: {modality} {data}')
            sample[modality] = data
        return sample


class ZeroMeanUnitVarianceRgbTensor:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, sample):
        if MOD_RGB not in sample:
            return sample
        assert torch.is_tensor(sample[MOD_RGB]), f'Invalid RGB modality type {type(sample[MOD_RGB])}'
        mean = torch.tensor(self.mean).view(3, 1, 1)
        stddev = torch.tensor(self.stddev).view(3, 1, 1)
        sample[MOD_RGB] = (sample[MOD_RGB] - mean) / stddev
        return sample


def get_transforms(
    semseg_ignore_class=None,
    geom_scale_min=0.5,
    geom_scale_max=2.0,
    geom_tilt_max_deg=0,
    geom_wiggle_max_ratio=0,
    geom_reflect=True,
    crop_for_passable=0,
    crop_random=0,
    crop_center=0,
    rgb_zero_mean_status=False,
    rgb_mean=None,
    rgb_stddev=None,
    scribble_shrink=1,
    stroke_width=1,
):
    return Compose([
        CropForPassableSidesPIL(crop_for_passable) if crop_for_passable > 0 else Identity(),
        RandomScaledTiltedWarpedCropPIL(
            crop_random,
            random_scale_min=geom_scale_min,
            random_scale_max=geom_scale_max,
            random_tilt_max_deg=geom_tilt_max_deg,
            random_wiggle_max_ratio=geom_wiggle_max_ratio,
            random_reflect=geom_reflect,
            semseg_ignore_class=semseg_ignore_class,
        ) if crop_random > 0 else Identity(),
        CropCenterPIL(
            crop_center,
            semseg_ignore_class=semseg_ignore_class,
        ) if crop_center > 0 else Identity(),
        ScribblesShrink(scribble_shrink) if 0 < scribble_shrink < 1 else Identity(),
        WeakRasterizerPIL(semseg_ignore_class, stroke_width),
        ConvertToTensorsSimple(),
        ZeroMeanUnitVarianceRgbTensor(rgb_mean, rgb_stddev) if rgb_zero_mean_status else Identity(),
    ])
