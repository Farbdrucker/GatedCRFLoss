#!/usr/bin/env python
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.logging import TestTubeLogger

from models.model_task_semseg_ce_gatedcrf import ModelTaskLightning
from utils.config import command_line_parser


def main():
    cfg = command_line_parser()

    model = ModelTaskLightning(cfg)

    logger = TestTubeLogger(
        save_dir=os.path.join(cfg.log_dir),
        name='tube'
    )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(cfg.log_dir, 'checkpoints'),
        save_best_only=True,
        verbose=True,
        monitor='_mean',
        mode='max',
        prefix=''
    )

    n_samples_in_ds = len(model.dataset_train)
    n_epochs = (cfg.num_batches_train_total * cfg.batch_size + n_samples_in_ds - 1) // n_samples_in_ds
    check_val_every_n_epoch = max(1, cfg.num_batches_validation_step * cfg.batch_size // n_samples_in_ds)

    trainer = Trainer(
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        default_save_path=os.path.join(cfg.log_dir, 'default'),
        gpus='-1',                  # all available GPUs
        show_progress_bar=True,
        check_val_every_n_epoch=check_val_every_n_epoch,
        fast_dev_run=cfg.__dict__.get('fast_dev_run', False),
        max_nb_epochs=n_epochs,
        log_save_interval=100,
        row_log_interval=1,
        distributed_backend='dp',   # single- or multi-gpu on a single node
        print_nan_grads=False,
        weights_summary='full',
        weights_save_path=None,
        nb_sanity_val_steps=1
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
