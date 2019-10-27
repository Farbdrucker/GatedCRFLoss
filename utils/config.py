import argparse
import os
import json


def expandpath(path):
    return os.path.expandvars(os.path.expanduser(path))


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def command_line_parser():
    parser = argparse.ArgumentParser(
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--log_dir', type=expandpath, required=True, default='.', help='Place for artifacts and logs')

    parser.add_argument(
        '--datasets_dir', type=expandpath, required=True, help='Path to dataset')
    parser.add_argument(
        '--dataset', type=str, required=True, choices=['voc', 'cs'], help='Pascal VOC or CityScapes')
    parser.add_argument(
        '--dataset_download', type=str2bool, default=False, help='Download dataset if possible')

    parser.add_argument(
        '--workers', type=int, default=16, help='Number of worker threads fetching training data')
    parser.add_argument(
        '--workers_validation', type=int, default=4, help='Number of worker threads fetching validation data')

    parser.add_argument(
        '--num_batches_train_total', type=int, default=90000, help='Number of training steps')
    parser.add_argument(
        '--num_batches_validation_step', type=int, default=5000, help='Number of steps between validations')

    parser.add_argument(
        '--batch_size', type=int, required=True, help='Number of samples in a batch for training')
    parser.add_argument(
        '--batch_size_validation', type=int, default=8, help='Number of samples in a batch for validation')

    parser.add_argument(
        '--aug_input_crop_size', type=int, required=True, help='Training crop size')
    parser.add_argument(
        '--aug_geom_scale_min', type=float, default=0.5, help='Augmentation: lower bound of scale')
    parser.add_argument(
        '--aug_geom_scale_max', type=float, default=2.0, help='Augmentation: upper bound of scale')
    parser.add_argument(
        '--aug_geom_tilt_max_deg', type=float, default=0.0, help='Augmentation: maximum rotation degree')
    parser.add_argument(
        '--aug_geom_wiggle_max_ratio', type=float, default=0.0,
        help='Augmentation: perspective warping level between 0 and 1')
    parser.add_argument(
        '--aug_geom_reflect', type=str2bool, default=True, help='Augmentation: Random horizontal flips')
    parser.add_argument(
        '--aug_geom_validation_center_crop_sts', type=str2bool, required=True,
        help='Augmentation: Enables center cropping during validation (useful for VOC)')
    parser.add_argument(
        '--aug_geom_validation_center_crop_size', type=int, default=512,
        help='Augmentation: Size of center crop during validation')
    parser.add_argument(
        '--aug_semseg_weak_stroke_width', type=int, default=1,
        help='Augmentation: Stroke width to use for rasterization of weak modalities')

    optimizer_kwargs_defaults = '{"lr": 0.007, "momentum": 0.9, "dampening": 0, "weight_decay": 0.0001}'
    parser.add_argument(
        '--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Type of optimizer')
    parser.add_argument(
        '--optimizer_kwargs', type=json.loads, default=optimizer_kwargs_defaults,
        help='Optimizer settings (defaults to DeepLab)')

    parser.add_argument(
        '--lr_scheduler', type=str, default='poly', choices=['poly'], help='Type of learning rate scheduler')
    parser.add_argument(
        '--lr_scheduler_power', type=float, default=0.9, help='Poly learning rate power')

    parser.add_argument(
        '--model_name', type=str, default='deeplabv3p', choices=['deeplabv3p'], help='CNN architecture')
    parser.add_argument(
        '--model_encoder_name', type=str, default='resnet34', choices=['resnet34', 'resnet50', 'resnet101'],
        help='CNN architecture encoder')

    parser.add_argument(
        '--loss_cross_entropy_class_weights_sts', type=str2bool, required=True,
        help='Enables class-weighted cross entropy')
    parser.add_argument(
        '--loss_cross_entropy_modality', type=str, required=True,
        choices=['semseg_dense', 'semseg_scribbles', 'semseg_clicks'],
        help='Which kind of (weak) supervision to use for cross entropy')
    parser.add_argument(
        '--loss_cross_entropy_weight', type=float, default=1.0, help='Cross entropy loss weight')

    loss_gatedcrf_kernels_desc_defaults = '[{"weight": 1, "xy": 6, "rgb": 0.1}]'
    parser.add_argument(
        '--loss_gatedcrf_sts', type=str2bool, required=True, help='Enables Gated CRF loss')
    parser.add_argument(
        '--loss_gatedcrf_resolution', type=int, required=True, help='Resolution on which Gated CRF is applied')
    parser.add_argument(
        '--loss_gatedcrf_radius', type=int, default=5, help='Radius of Gated CRF kernels')
    parser.add_argument(
        '--loss_gatedcrf_kernels_desc', type=json.loads, default=loss_gatedcrf_kernels_desc_defaults,
        help='Descriptor of Gated CRF kernels')
    parser.add_argument(
        '--loss_gatedcrf_weight', type=float, default=0.1, help='Gated CRF loss weight')
    parser.add_argument(
        '--loss_gatedcrf_use_after_progress_ratio', type=float, default=0.005,
        help='Gated CRF loss idle time relative to whole experiment')

    parser.add_argument(
        '--num_batches_visualization_first', type=int, default=100, help='Visualization: first time step')
    parser.add_argument(
        '--num_batches_visualization_step', type=int, default=1000, help='Visualization: interval in steps')
    parser.add_argument(
        '--visualize_num_samples_in_batch', type=int, default=8, help='Visualization: max number of samples in batch')
    parser.add_argument(
        '--observe_train_ids', type=json.loads, default='[0,100]', help='Visualization: train IDs')
    parser.add_argument(
        '--observe_valid_ids', type=json.loads, default='[0,100]', help='Visualization: validation IDs')
    parser.add_argument(
        '--tensorboard_img_grid_width', type=int, default=8, help='Visualization: number of samples per row')

    cfg = parser.parse_args()

    print(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

    return cfg
