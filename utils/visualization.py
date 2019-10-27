import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.font_manager import findfont, FontProperties
from PIL import Image, ImageFont, ImageDraw
from torchvision.utils import make_grid

from datasets.definitions import MOD_SS_CLICKS, MOD_RGB, MOD_SS_DENSE, MOD_VALIDITY, MOD_SS_SCRIBBLES


class ImageTextRenderer:
    def __init__(self, size=60):
        font_path = findfont(FontProperties(family='monospace'))
        self.font = ImageFont.truetype(font_path, size=size, index=0)
        self.size = size

    def print_gray(self, img_np_f, text, offs_xy, white=1.0):
        assert len(img_np_f.shape) == 2, "Image must be single channel"
        img_pil = Image.fromarray(img_np_f, mode='F')
        ctx = ImageDraw.Draw(img_pil)
        step = self.size // 15
        for dx, dy in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
            ctx.text((offs_xy[0] + step * dx, offs_xy[1] + step * dy), text, font=self.font, fill=0.0)
        ctx.text(offs_xy, text, font=self.font, fill=white)
        return np.array(img_pil)

    def print(self, img_np_f, text, offs_xy, **kwargs):
        if len(img_np_f.shape) == 3:
            for ch in range(3):
                img_np_f[ch] = self.print_gray(img_np_f[ch], text, offs_xy, **kwargs)
        else:
            img_np_f = self.print_gray(img_np_f, text, offs_xy, **kwargs)
        return img_np_f


_text_renderers = dict()


def get_text_renderer(size):
    if size not in _text_renderers:
        _text_renderers[size] = ImageTextRenderer(size)
    return _text_renderers[size]


def img_print(*args, **kwargs):
    size = kwargs['size']
    del kwargs['size']
    renderer = get_text_renderer(size)
    return renderer.print(*args, **kwargs)


def tensor_print(img, caption, **kwargs):
    if isinstance(caption, str) and len(caption.strip()) == 0:
        return img
    assert img.dim() == 4 and img.shape[1] in (1, 3), 'Expecting 4D tensor with RGB or grayscale'
    offset = min(img.shape[2], img.shape[3]) // 100
    img = img.cpu()
    offset = (offset, offset)
    if 'offsetx' in kwargs.keys():
        offset = (kwargs['offsetx'], kwargs['offsety'])
        del kwargs['offsetx'], kwargs['offsety']
    size = min(img.shape[2], img.shape[3]) // 15
    for i in range(img.shape[0]):
        tag = (caption if isinstance(caption, str) else caption[i]).strip()
        if len(tag) == 0:
            continue
        img_np = img_print(img[i].numpy(), tag, offset, size=size, **kwargs)
        img[i] = torch.from_numpy(img_np)
    return img


def prepare_rgb(img, rgb_mean, rgb_stddev):
    assert img.dim() == 4 and img.shape[1] == 3, 'Expecting 4D tensor with RGB'
    img = img.float().cpu()
    img_mean, img_stddev = torch.tensor(rgb_mean).float().view(3, 1, 1), torch.tensor(rgb_stddev).float().view(3, 1, 1)
    return torch.clamp((img * img_stddev + img_mean) / 255, 0, 1)


def superimpose_rgb(img, rgb, opacity=0.5):
    gray = rgb2gray(rgb)
    img = (1 - opacity) * gray + opacity * img
    return img


def rgb2gray(rgb):
    return rgb[:, 1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)  # RGB -> GGG


def create_checkerboard(N, C, H, W):
    cell_sz = max(min(H, W) // 32, 1)
    mH = (H + cell_sz - 1) // cell_sz
    mW = (W + cell_sz - 1) // cell_sz
    checkerboard = torch.full((mH, mW), 0.25, dtype=torch.float32)
    checkerboard[0::2, 0::2] = 0.75
    checkerboard[1::2, 1::2] = 0.75
    checkerboard = checkerboard.float().view(1, 1, mH, mW)
    checkerboard = F.interpolate(checkerboard, scale_factor=cell_sz, mode='nearest')
    checkerboard = checkerboard[:, :, :H, :W].repeat(N, C, 1, 1)
    return checkerboard


def prepare_semseg(img, semseg_color_map, semseg_ignore_class, rgb=None):
    assert img.dim() == 4 and img.shape[1] == 1 and img.dtype in (torch.int, torch.long), \
        f'Expecting 4D tensor with semseg classes, got {img.shape}'
    colors = torch.tensor(semseg_color_map, dtype=torch.float32)
    assert colors.dim() == 2 and colors.shape[1] == 3
    if torch.max(colors) > 128:
        colors /= 255
    img = img.cpu().squeeze(1).clone()  # N x H x W
    N, H, W = img.shape
    img_color_ids = torch.unique(img)
    assert all(c_id == semseg_ignore_class or 0 <= c_id < len(semseg_color_map) for c_id in img_color_ids)
    checkerboard, mask_ignore = None, None
    if semseg_ignore_class in img_color_ids:
        if rgb is None:
            checkerboard = create_checkerboard(N, 3, H, W)
        mask_ignore = img == semseg_ignore_class
        img[mask_ignore] = 0
    img = colors[img]  # N x H x W x 3
    img = img.permute(0, 3, 1, 2)
    if semseg_ignore_class in img_color_ids:
        mask_ignore = mask_ignore.unsqueeze(1).repeat(1, 3, 1, 1)
        img[mask_ignore] = (checkerboard if rgb is None else rgb2gray(rgb * 0.5))[mask_ignore]
    return img


def prepare_semseg_clicks(img, semseg_color_map, semseg_ignore_class, rgb):
    assert img.dim() == 4 and img.shape[1] == 1 and img.dtype in (torch.int, torch.long), \
        f'Expecting 4D tensor with semseg classes, got {img.shape}'
    rgb = rgb2gray(rgb) * 255
    img = img.cpu().squeeze(1)  # N x H x W
    N, H, W = img.shape
    img_color_ids = torch.unique(img)
    assert all(c_id == semseg_ignore_class or 0 <= c_id < len(semseg_color_map) for c_id in img_color_ids)
    mask = img != semseg_ignore_class
    clicks_x = torch.arange(W, dtype=torch.long, device=img.device).view(1, 1, W)
    clicks_y = torch.arange(H, dtype=torch.long, device=img.device).view(1, H, 1)
    clicks_n = torch.arange(N, dtype=torch.long, device=img.device).view(N, 1, 1)
    clicks_clsid = torch.masked_select(img, mask).tolist()
    clicks_x = torch.masked_select(clicks_x, mask).tolist()
    clicks_y = torch.masked_select(clicks_y, mask).tolist()
    clicks_n = torch.masked_select(clicks_n, mask).tolist()
    img = [Image.fromarray(rgb[n, :, :, :].permute(1, 2, 0).to(torch.uint8).numpy()) for n in range(N)]
    img_draw = [ImageDraw.Draw(i) for i in img]
    rad_click = min(W, H) // 64
    rad_outline = rad_click * 1.5
    for n, y, x, clsid in zip(clicks_n, clicks_y, clicks_x, clicks_clsid):
        x, y = float(x) + 0.5, float(y) + 0.5
        color_rgb = semseg_color_map[clsid]
        color_outline = (0, 0, 0) if sum(color_rgb) > 3 * 128 else (255, 255, 255)
        img_draw[n].ellipse((x - rad_outline, y - rad_outline, x + rad_outline, y + rad_outline), fill=color_outline, outline=color_outline)
        img_draw[n].ellipse((x - rad_click, y - rad_click, x + rad_click, y + rad_click), fill=color_rgb, outline=color_rgb)
    del img_draw
    img = [torch.from_numpy(np.array(i)).permute(2, 0, 1).unsqueeze(0).float() / 255 for i in img]
    img = torch.cat(img, dim=0)
    return img


def prepare_mask(mask):
    assert mask.dim() == 4 and mask.shape[1] == 1
    mask = mask.float().cpu()
    out = torch.cat((mask, mask, mask), dim=1)
    return out


def save_indexed_segmap(filepath, img, semseg_color_map, semseg_ignore_class=None, semseg_ignore_color=(0, 0, 0)):
    assert torch.is_tensor(img)
    img = img.squeeze()
    assert img.dim() == 2 and img.dtype in (torch.int, torch.long), 'Expecting 2D tensor with semseg classes'
    img = img.cpu().byte().numpy()
    img_pil = Image.fromarray(img, mode='P')
    palette = [0 for _ in range(256 * 3)]
    for i, rgb in enumerate(semseg_color_map):
        for c in range(3):
            palette[3 * i + c] = rgb[c]
    if semseg_ignore_class is not None:
        for c in range(3):
            palette[3 * semseg_ignore_class + c] = semseg_ignore_color[c]
    img_pil.putpalette(palette)
    img_pil.save(filepath)


def compose(list_triples, cfg, rgb_mean=None, rgb_stddev=None, semseg_color_map=None, semseg_ignore_class=None):
    if cfg.visualize_num_samples_in_batch is not None:
        list_triples = [(m, img[:cfg.visualize_num_samples_in_batch], c) for (m, img, c) in list_triples]
    N, H, W, rgb = None, None, None, None
    for (modality, img, caption) in list_triples:
        if modality == MOD_RGB:
            N, _, H, W = img.shape
            rgb = prepare_rgb(img, rgb_mean, rgb_stddev)
            break
    vis = []
    for i, (modality, img, caption) in enumerate(list_triples):
        if modality == MOD_RGB:
            vis.append(tensor_print(
                rgb.clone(), caption if type(caption) is list else [str(idx) for idx in caption.tolist()])
            )
        elif modality == MOD_SS_DENSE:
            vis.append(tensor_print(prepare_semseg(img, semseg_color_map, semseg_ignore_class), caption))
        elif modality == MOD_SS_SCRIBBLES:
            vis.append(tensor_print(prepare_semseg(img, semseg_color_map, semseg_ignore_class, rgb), caption))
        elif modality == MOD_SS_CLICKS:
            vis.append(tensor_print(prepare_semseg_clicks(img, semseg_color_map, semseg_ignore_class, rgb), caption))
        elif modality == MOD_VALIDITY:
            vis.append(tensor_print(prepare_mask(img), caption))
    vis = torch.cat(vis, dim=2)
    # N x 3 x H * N_MODS x W
    vis = make_grid(vis, nrow=min(N, cfg.tensorboard_img_grid_width))
    return vis
