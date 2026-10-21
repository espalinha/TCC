"""Etapa 2: geracao 2D multi-vista com TeethDreamer."""

from commands import CommandRunner
from pipeline_config import PathLayout, PipelineConfig


class TeethDreamerStep:
    name = "Geracao 2D multi-vista (TeethDreamer)"

    def __init__(self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def _prepare_staging(self) -> None:
        """Cria symlinks <uid>_upper / <uid>_lower em staging_dir."""
        raise NotImplementedError
