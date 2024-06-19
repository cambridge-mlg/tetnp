import wandb
from plot import plot
from plot_image import plot_image
from tnp.utils.experiment_utils import initialize_evaluation

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
            subplots=experiment.misc.subplots,
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging,
        )
    else:
        plot(
            model=model,
            batches=batches,
            num_fig=min(experiment.misc.num_plots, len(batches)),
            name=eval_name,
            savefig=experiment.misc.savefig,
            logging=experiment.misc.logging,
        )


if __name__ == "__main__":
    main()
