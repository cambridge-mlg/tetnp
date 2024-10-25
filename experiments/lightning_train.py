import os

import lightning.pytorch as pl
import torch
import wandb
from omegaconf import OmegaConf
from plot import plot
from plot_era5 import plot_era5
from plot_image import plot_image
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper, LogPerformanceCallback

from tetnp.data.era5 import ERA5DataGenerator
from tetnp.data.image import ImageGenerator


def main():
    experiment = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    train_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_workers > 0 else False,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        gen_train,
        batch_size=None,
        num_workers=experiment.misc.num_val_workers,
        worker_init_fn=(
            (
                experiment.misc.worker_init_fn
                if hasattr(experiment.misc, "worker_init_fn")
                else adjust_num_batches
            )
            if experiment.misc.num_val_workers > 0
            else None
        ),
        persistent_workers=True if experiment.misc.num_val_workers > 0 else False,
        pin_memory=False,
    )

    def plot_fn(model, batches, name):
        if isinstance(gen_train, ImageGenerator):
            plot_image(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )
        elif isinstance(gen_train, ERA5DataGenerator):
            plot_era5(
                model=model,
                batches=batches,
                x_mean=gen_val.x_mean,
                x_std=gen_val.x_std,
                y_mean=gen_val.y_mean,
                y_std=gen_val.y_std,
                num_fig=min(5, len(batches)),
                lat_range=gen_val.lat_range,
                lon_range=gen_val.lon_range,
                time_idx=[0, -1],
                name=name,
                use_time=gen_val.use_time,
            )
        else:
            plot(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                name=name,
            )

    if experiment.misc.resume_from_checkpoint is not None:
        api = wandb.Api()
        artifact = api.artifact(experiment.misc.resume_from_checkpoint)
        artifact_dir = artifact.download()
        ckpt_file = os.path.join(artifact_dir, "model.ckpt")

        lit_model = (
            LitWrapper.load_from_checkpoint(  # pylint: disable=no-value-for-parameter
                ckpt_file,
            )
        )
    else:
        ckpt_file = None
        lit_model = LitWrapper(
            model=model,
            optimiser=optimiser,
            loss_fn=experiment.misc.loss_fn,
            pred_fn=experiment.misc.pred_fn,
            plot_fn=plot_fn,
            plot_interval=experiment.misc.plot_interval,
        )

    if experiment.misc.logging:
        logger = pl.loggers.WandbLogger(
            project=experiment.misc.project,
            name=experiment.misc.name,
            config=OmegaConf.to_container(experiment.config),
            log_model="all",
        )
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            every_n_epochs=experiment.misc.checkpoint_interval,
            save_last=True,
        )
        performance_callback = LogPerformanceCallback()
        callbacks = [checkpoint_callback, performance_callback]
    else:
        logger = False
        callbacks = None

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=(
            experiment.misc.log_interval if not experiment.misc.logging else None
        ),
        devices=1,
        accelerator="cpu",
        num_sanity_val_steps=1,
        check_val_every_n_epoch=(experiment.misc.check_val_every_n_epoch),
        gradient_clip_val=experiment.misc.gradient_clip_val,
        callbacks=callbacks,
    )

    trainer.fit(
        model=lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=ckpt_file,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
