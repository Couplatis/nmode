import torch
import typer
import rich

from nmode.trainer import CIFAR10Trainer, MNISTTrainer

app = typer.Typer(
    name="nmode",
    help="Neural Memory ODE Command Line Interface",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def main(
    dataset: str = typer.Option(default="MNIST", help="Dataset to train on."),
    device: torch.device = typer.Option(
        default=torch.device(torch.cuda.is_available() and "cuda" or "cpu"),
        parser=lambda x: torch.device(x),
        help="Device to train on.",
    ),
    epochs: int = typer.Option(default=10, help="Number of epochs to train for."),
    batch_size: int = typer.Option(default=256, help="Batch size to use."),
):
    if dataset not in ["MNIST", "CIFAR10"]:
        typer.echo("Invalid dataset.")
        raise typer.Exit(code=1)

    rich.print(
        f"[bold]Neural Memory ODE[/bold] is training on [italic green]{device}[/italic green],",
        f"for [italic green]{epochs}[/italic green] epochs,",
        f"with a batch size of [italic green]{batch_size}[/italic green].",
    )
    rich.print(f"Using [italic green]{dataset}[/italic green] dataset.")

    if dataset == "MNIST":
        trainer = MNISTTrainer(device)
    else:
        trainer = CIFAR10Trainer(device)

    trainer.train(epochs, batch_size)


if __name__ == "__main__":
    app()
