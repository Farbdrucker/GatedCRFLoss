from math import nan

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from models.model_loss_semseg_gatedcrf import ModelLossSemsegGatedCRF
from datasets.definitions import SPLIT_TRAIN, SPLIT_VALID, MOD_RGB, MOD_SS_DENSE, MOD_VALIDITY, MOD_ID
from utils.helpers import resolve_optimizer, resolve_dataset_class, resolve_network_model, PolyLR
from utils.semseg import class_weights_from_histogram, semseg_compute_confusion, semseg_accum_confusion_to_iou
from utils.transforms import get_transforms
from utils.visualization import compose


class ModelTaskLightning(pl.LightningModule):
    def __init__(self, cfg):
        super(ModelTaskLightning, self).__init__()
        assert cfg.lr_scheduler == 'poly', 'This implementation relies on an unconventional usage of _LRScheduler class'
        self.cfg = cfg

        dataset_class = resolve_dataset_class(cfg.dataset)

        self.dataset_train = dataset_class(
            cfg.datasets_dir, SPLIT_TRAIN, download=cfg.dataset_download, integrity_check=False
        )
        self.dataset_valid = dataset_class(
            cfg.datasets_dir, SPLIT_VALID, download=cfg.dataset_download, integrity_check=False
        )

        print('Number of samples in training split:', len(self.dataset_train))
        print('Number of samples in validation split:', len(self.dataset_valid))

        self.semseg_num_classes = self.dataset_train.num_classes
        self.semseg_ignore_class = self.dataset_train.ignore_label
        self.semseg_class_names = self.dataset_train.semseg_class_names

        model_class = resolve_network_model(cfg.model_name)
        self.net = model_class(cfg, self.semseg_num_classes)

        self.transforms_train = get_transforms(
            semseg_ignore_class=self.semseg_ignore_class,
            geom_scale_min=cfg.aug_geom_scale_min,
            geom_scale_max=cfg.aug_geom_scale_max,
            geom_tilt_max_deg=cfg.aug_geom_tilt_max_deg,
            geom_wiggle_max_ratio=cfg.aug_geom_wiggle_max_ratio,
            geom_reflect=cfg.aug_geom_reflect,
            crop_random=cfg.aug_input_crop_size,
            rgb_zero_mean_status=True,
            rgb_mean=self.dataset_train.rgb_mean,
            rgb_stddev=self.dataset_train.rgb_stddev,
            stroke_width=cfg.aug_semseg_weak_stroke_width,
        )

        self.transforms_valid = get_transforms(
            semseg_ignore_class=self.semseg_ignore_class,
            crop_for_passable=self.net.bottleneck_stride if not cfg.aug_geom_validation_center_crop_sts else 0,
            crop_center=cfg.aug_geom_validation_center_crop_size if cfg.aug_geom_validation_center_crop_sts else 0,
            rgb_zero_mean_status=True,
            rgb_mean=self.dataset_train.rgb_mean,
            rgb_stddev=self.dataset_train.rgb_stddev,
            stroke_width=cfg.aug_semseg_weak_stroke_width,
        )

        self.dataset_train.set_transforms(self.transforms_train)
        self.dataset_valid.set_transforms(self.transforms_valid)

        class_weights = None
        if cfg.loss_cross_entropy_class_weights_sts:
            class_weights = class_weights_from_histogram(self.dataset_train.semseg_class_histogram)

        self.loss_ce = torch.nn.CrossEntropyLoss(class_weights, ignore_index=self.semseg_ignore_class)

        self.loss_gatedcrf = None
        if cfg.loss_gatedcrf_sts:
            self.loss_gatedcrf = ModelLossSemsegGatedCRF()

        self.poly_lr_sched = None

    def training_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y_hat = self.net(rgb)
        y_ce = batch[self.cfg.loss_cross_entropy_modality]
        loss_ce = self.loss_ce(y_hat['high'], y_ce.squeeze(1))

        N, _, H, W = rgb.shape

        visualization_plan = []
        is_vis_step = self.can_visualize() and \
                      (self.global_step - self.cfg.num_batches_visualization_first) % self.cfg.num_batches_visualization_step == 0
        if is_vis_step:
            self.visualization_common_part(visualization_plan, y_hat['high'], rgb, batch[MOD_ID], batch)

        loss_denom = self.cfg.loss_cross_entropy_weight
        loss_total = loss_ce * self.cfg.loss_cross_entropy_weight
        tensorboard_logs = {
            'loss/cross_entropy': loss_ce.unsqueeze(0),
        }

        progress = self.global_step / self.cfg.num_batches_train_total
        assert 0 <= progress <= 1
        gatedcrf_progress_threshold = self.cfg.loss_gatedcrf_use_after_progress_ratio

        if self.cfg.loss_gatedcrf_sts and progress > gatedcrf_progress_threshold:
            validity = batch[MOD_VALIDITY]

            y_hat_softmax = y_hat['low'].softmax(dim=1)

            out_gatedcrf = self.loss_gatedcrf(
                y_hat_softmax,
                self.cfg.loss_gatedcrf_kernels_desc,
                self.cfg.loss_gatedcrf_radius,
                batch,
                rgb.shape[2],
                rgb.shape[3],
                mask_src=validity,
                out_kernels_vis=is_vis_step,
            )

            loss_gatedcrf = out_gatedcrf['loss']

            loss_total = loss_total + loss_gatedcrf * self.cfg.loss_gatedcrf_weight
            loss_denom += self.cfg.loss_gatedcrf_weight
            tensorboard_logs['loss/gatedcrf'] = loss_gatedcrf.unsqueeze(0)

            if is_vis_step:
                visualization_plan.append((MOD_VALIDITY, out_gatedcrf['kernels_vis'], 'Kernels'))
                visualization_plan.append((MOD_VALIDITY, validity, 'Validity'))

        if is_vis_step:
            self.visualization_submit_plan(visualization_plan, 'train', rgb.device)

        tensorboard_logs['loss/_total'] = loss_total.unsqueeze(0) / loss_denom

        lrs = self.poly_lr_sched.get_lr()
        tensorboard_logs['LR'] = torch.tensor([lrs[0]], device=loss_total.device)

        return {'loss': loss_total.unsqueeze(0), 'log': tensorboard_logs}

    def on_batch_end(self):
        # semantically this belongs to on_before_zero_grad, which is never called for some reasonw
        self.poly_lr_sched.step(self.global_step)

    def validation_step(self, batch, batch_nb):
        rgb = batch[MOD_RGB]
        y = batch[MOD_SS_DENSE]
        y_hat = self.net(rgb)['high']
        y_hat_lbl = y_hat.argmax(dim=1, keepdim=True)
        confusion = semseg_compute_confusion(y_hat_lbl, y, self.semseg_num_classes, self.semseg_ignore_class)
        return confusion.unsqueeze(0)

    def validation_end(self, outputs):
        outputs = (confusion.sum(dim=0) for confusion in outputs)
        outputs = sum(outputs)
        iou_mean, iou_per_class = semseg_accum_confusion_to_iou(outputs)
        out = {self.semseg_class_names[i]: iou for i, iou in enumerate(iou_per_class)}
        out['_mean'] = iou_mean
        self.observer_step()
        return {
            'progress_bar': {'mIoU': iou_mean},
            'log': {f'IoU/{k}': v for k, v in out.items()},
            '_mean': iou_mean
        }

    def configure_optimizers(self):
        optimizer_class = resolve_optimizer(self.cfg.optimizer)
        optimizer = optimizer_class(self.parameters(), **self.cfg.optimizer_kwargs)
        # The framework allows to step LR only once per epoch - if returning LR scheduler along with the optimizer;
        # However, our training protocol requires LR steps after every batch, hence we manually load/store, and overload
        # on_batch_end to pass global_step instead of epoch
        self.poly_lr_sched = PolyLR(optimizer, self.cfg.lr_scheduler_power, self.cfg.num_batches_train_total)
        return optimizer

    def on_load_checkpoint(self, checkpoint):
        self.poly_lr_sched.load_state_dict(checkpoint['poly_lr_sched'])

    def on_save_checkpoint(self, checkpoint):
        checkpoint['poly_lr_sched'] = self.poly_lr_sched.state_dict()

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.workers,
            pin_memory=True,
            drop_last=True,
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.dataset_valid,
            self.cfg.batch_size_validation,
            shuffle=False,
            num_workers=self.cfg.workers_validation,
            pin_memory=True,
            drop_last=False,
        )

    def visualization_common_part(self, visualization_plan, y_hat, rgb, rgb_tags, batch):
        y_hat_lbl = y_hat.argmax(dim=1, keepdim=True)
        visualization_plan.append((MOD_RGB, rgb, rgb_tags))
        visualization_plan.append((MOD_SS_DENSE, y_hat_lbl, 'Prediction'))
        if self.cfg.loss_cross_entropy_modality == MOD_SS_DENSE:
            visualization_plan.append((MOD_SS_DENSE, batch[MOD_SS_DENSE], 'Supervision w/ Dense GT'))
        else:
            visualization_plan.append(
                (self.cfg.loss_cross_entropy_modality, batch[self.cfg.loss_cross_entropy_modality], 'Supervision')
            )
            visualization_plan.append((MOD_SS_DENSE, batch[MOD_SS_DENSE], 'Dense GT'))

    def visualization_submit_plan(self, visualization_plan, prefix, device):
        vis = compose(
            visualization_plan, self.cfg,
            rgb_mean=self.dataset_train.rgb_mean,
            rgb_stddev=self.dataset_train.rgb_stddev,
            semseg_color_map=self.dataset_train.semseg_class_colors,
            semseg_ignore_class=self.dataset_train.ignore_label,
        )
        self.logger.experiment.add_image(f'{prefix}/{device}', vis, self.global_step)

    @staticmethod
    def can_visualize():
        # prevent multiple GPU workers from doing polluting TF log db
        return torch.cuda.current_device() == 0

    def observer_step(self):
        if not self.can_visualize():
            return
        vis_transforms = self.transforms_valid
        list_samples = []
        for i in self.cfg.observe_train_ids:
            list_samples.append(self.dataset_train.get(i, override_transforms=vis_transforms))
        for i in self.cfg.observe_valid_ids:
            list_samples.append(self.dataset_valid.get(i, override_transforms=vis_transforms))
        list_prefix = ('train/', ) * len(self.cfg.observe_train_ids) + ('valid/', ) * len(self.cfg.observe_valid_ids)
        batch = default_collate(list_samples)
        rgb = batch[MOD_RGB]
        rgb_tags = [f'{prefix}{id}' for prefix, id in zip(list_prefix, batch[MOD_ID])]
        with torch.no_grad():
            y_hat = self.net(rgb.cuda())['high'].cpu()
        visualization_plan = []
        self.visualization_common_part(visualization_plan, y_hat, rgb, rgb_tags, batch)
        self.visualization_submit_plan(visualization_plan, 'valid', 'cpu')
