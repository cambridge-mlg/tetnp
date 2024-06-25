import wandb
from plot import plot
from plot_era5 import plot_era5
from plot_image import plot_image
from tnp.utils.experiment_utils import initialize_evaluation

from tetnp.data.era5 import ERA5DataGenerator
from tetnp.data.image import ImageGenerator


def main():
    experiment = initialize_evaluation()

    model = experiment.model
    eval_name = experiment.misc.eval_name
    gen_test = experiment.generators.test

    model.eval()

    gen_test.batch_size = 1
    gen_test.num_batches = experiment.misc.num_plots
    batches = list(iter(gen_test))

    eval_name = wandb.run.name + "/" + eval_name

    if isinstance(gen_test, ImageGenerator):
        plot_image(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            figsize=(6, 6),
            name=eval_name,
            savefig=(
                False
                if not hasattr(experiment.misc.savefig)
                else experiment.misc.savefig
            ),
            logging=(
                False
                if not hasattr(experiment.misc.logging)
                else experiment.misc.logging
            ),
        )
    elif isinstance(gen_test, ERA5DataGenerator):
        plot_era5(
            model=model,
            batches=batches,
            x_mean=gen_test.x_mean,
            x_std=gen_test.x_std,
            y_mean=gen_test.y_mean,
            y_std=gen_test.y_std,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            figsize=(24.0, 5.0),
            lat_range=gen_test.lat_range,
            lon_range=gen_test.lon_range,
            time_idx=[0, -1],
            name=eval_name,
            subplots=(
                True
                if not hasattr(experiment.misc.subplots)
                else experiment.misc.subplots
            ),
            savefig=(
                False
                if not hasattr(experiment.misc.savefig)
                else experiment.misc.savefig
            ),
            logging=(
                False
                if not hasattr(experiment.misc.logging)
                else experiment.misc.logging
            ),
            use_time=gen_test.use_time,
        )
    else:
        plot(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name=eval_name,
            savefig=(
                False
                if not hasattr(experiment.misc.savefig)
                else experiment.misc.savefig
            ),
            logging=(
                False
                if not hasattr(experiment.misc.logging)
                else experiment.misc.logging
            ),
        )


if __name__ == "__main__":
    main()
