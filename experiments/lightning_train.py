import lightning.pytorch as pl
from plot import plot
from plot_era5 import plot_era5
from plot_image import plot_image
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper

from tetnp.data.era5 import ERA5DataGenerator
from tetnp.data.image import ImageGenerator


def main():
    experiment, checkpointer = initialize_experiment()

    model = experiment.model
    gen_train = experiment.generators.train
    gen_val = experiment.generators.val
    optimiser = experiment.optimiser(model.parameters())
    epochs = experiment.params.epochs

    def plot_fn(model, batches, name):
        if isinstance(gen_train, ImageGenerator):
            plot_image(
                model=model,
                batches=batches,
                num_fig=min(5, len(batches)),
                figsize=(6, 6),
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
                figsize=(24.0, 5.0),
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

    lit_model = LitWrapper(
        model=model,
        optimiser=optimiser,
        loss_fn=experiment.misc.loss_fn,
        pred_fn=experiment.misc.pred_fn,
        plot_fn=plot_fn,
        checkpointer=checkpointer,
        plot_interval=experiment.misc.plot_interval,
    )
    logger = pl.loggers.WandbLogger() if experiment.misc.logging else False
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=epochs,
        limit_train_batches=gen_train.num_batches,
        limit_val_batches=gen_val.num_batches,
        log_every_n_steps=1,
        devices=1,
        gradient_clip_val=experiment.misc.gradient_clip_val,
        accelerator="cpu",
    )

    trainer.fit(model=lit_model, train_dataloaders=gen_train, val_dataloaders=gen_val)


if __name__ == "__main__":
    main()
