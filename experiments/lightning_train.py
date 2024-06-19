import lightning.pytorch as pl
from plot import plot
from plot_image import plot_image
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.lightning_utils import LitWrapper

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
