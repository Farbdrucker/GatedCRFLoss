from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler

from datasets.dataset_cityscapes_with_weak import DatasetCityscapesWithWeak
from datasets.dataset_voc_sbd_with_weak import DatasetVocSbdWithWeak
from models.model_net_deeplab_v3_plus import ModelNetDeepLabV3Plus


def resolve_dataset_class(name):
    return {
        'voc': DatasetVocSbdWithWeak,
        'cs': DatasetCityscapesWithWeak,
    }[name]


def resolve_network_model(name):
    return {
        'deeplabv3p': ModelNetDeepLabV3Plus
    }[name]


def resolve_optimizer(name):
    return {
        'sgd': SGD,
        'adam': Adam,
    }[name]


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, power, num_steps, last_epoch=-1):
        self.power = power
        self.num_steps = num_steps
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0 - min(self.last_epoch, self.num_steps-1) / self.num_steps) ** self.power
                for base_lr in self.base_lrs]
