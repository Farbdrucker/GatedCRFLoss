import torch
import torch.nn.functional as F


def class_weights_from_histogram(hist, use_logp=True):
    if not torch.is_tensor(hist):
        hist = torch.tensor(hist)
    hist = hist.double()
    hist = F.normalize(hist, p=1, dim=0)
    if use_logp:
        # https://arxiv.org/pdf/1809.09077v1.pdf formula (3)
        log_safety_const = 1.10
        classes_weights = 1.0 / torch.log(hist + log_safety_const)
    else:
        classes_weights = 1.0 / hist.clamp(min=1e-6)
    classes_weights = torch.nn.functional.normalize(classes_weights, p=1, dim=0)
    return classes_weights.float()


def semseg_compute_confusion(y_hat_lbl, y, C, semseg_ignore_class):
    assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y), 'Inputs must be torch tensors'
    assert y.device == y_hat_lbl.device, 'Input tensors have different device placement'

    if y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1:
        y_hat_lbl = y_hat_lbl.squeeze(1)
    if y.dim() == 4 and y.shape[1] == 1:
        y = y.squeeze(1)

    mask = y != semseg_ignore_class
    y_hat_lbl = y_hat_lbl[mask]
    y = y[mask]

    # hack for bincounting 2 arrays together
    x = y_hat_lbl + C * y
    bincount_2d = torch.bincount(x.long(), minlength=C ** 2)
    assert bincount_2d.numel() == C ** 2, 'Internal error'
    conf = bincount_2d.view((C, C)).long()
    return conf


def semseg_accum_confusion_to_iou(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    iou_per_class = diag / (conf.sum(dim=1) + conf.sum(dim=0) - diag).clamp(min=1e-12)
    iou_mean = iou_per_class.mean()
    return iou_mean, iou_per_class
