from typing import List

from commands import CommandRunner
from pipeline_config import PathLayout, PipelineConfig


class ReconstructionStep:
    name = "Reconstrucao 3D (instant-nsr-pl)"

    def __init__(
        self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner
    ) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def _find_strips(self) -> List[str]:
        """Prefere *_nobg.png; senao PNGs com '_cond_'."""
        raise NotImplementedError

    @staticmethod
    def _arch_of(strip_name: str) -> str:
        """'lower' ou 'upper' a partir do nome do arquivo."""
        raise NotImplementedError
