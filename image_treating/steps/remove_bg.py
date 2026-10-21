"""Etapa 3: remocao de fundo branco dos strips gerados."""

from typing import List

from commands import CommandRunner
from pipeline_config import PathLayout, PipelineConfig


class RemoveBackgroundStep:
    name = "Remocao de fundo branco (remove_bg)"

    def __init__(self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def _pending_strips(self) -> List[str]:
        """PNGs de td_output_dir que ainda nao tem versao _nobg."""
        raise NotImplementedError
