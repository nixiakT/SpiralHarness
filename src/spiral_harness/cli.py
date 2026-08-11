"""Command-line entry point for SpiralHarness."""

from typing import Annotated

import typer

from spiral_harness import __version__

app = typer.Typer(
    name="spiral",
    help="Evidence-gated, training-free evolution for agent harnesses.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Run SpiralHarness commands."""


if __name__ == "__main__":  # pragma: no cover
    app()
