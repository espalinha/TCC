import os
import sys
from typing import List

from commands import CommandRunner
from errors import PipelineError
from pipeline_config import HERE, PathLayout, PipelineConfig


class RemoveBackgroundStep:
    name = "Remocao de fundo branco (remove_bg)"

    def __init__(self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner) -> None:
        self._layout, self._runner = layout, runner

    def run(self) -> None:
        td = self._layout.td_output_dir
        for fn in self._pending_strips():
            name, ext = os.path.splitext(fn)
            dst = os.path.join(td, f"{name}_nobg{ext}")
            cmd = [sys.executable, "-m", "fase2_geracao3d.fundo_branco", os.path.join(td, fn), dst]
            self._runner.run(cmd, cwd=HERE)

    def _pending_strips(self) -> List[str]:
        """PNGs de td_output_dir que ainda nao tem versao _nobg."""
        td = self._layout.td_output_dir
        if not os.path.isdir(td):
            raise PipelineError(f"teethdreamer_output nao encontrado: {td}")
        return sorted(f for f in os.listdir(td) if f.endswith(".png") and not f.endswith("_nobg.png"))
