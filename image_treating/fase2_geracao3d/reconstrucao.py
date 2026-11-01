import os
import sys
from typing import List

from commands import CommandRunner
from errors import PipelineError
from pipeline_config import PathLayout, PipelineConfig
from fase2_geracao3d.malhas import arch_of


class ReconstructionStep:
    name = "Reconstrucao 3D (instant-nsr-pl)"

    def __init__(
        self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner
    ) -> None:
        self._config, self._layout, self._runner = config, layout, runner

    def run(self) -> None:
        c, l = self._config, self._layout
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=c.gpu_id)
        for fn in self._find_strips():
            arch_dir = os.path.join(l.recon_dir, arch_of(fn))
            os.makedirs(arch_dir, exist_ok=True)
            cmd = [
                sys.executable, os.path.join(l.nsr_dir, "run.py"),
                "--img", os.path.join(l.td_output_dir, fn),
                "--cpu", str(c.cpu),
                "--dir", arch_dir,
                "--normal",
                "--gpu", c.gpu_id,
            ]
            if c.neus_config:
                cmd += ["--config", c.neus_config]
            self._runner.run(cmd, cwd=l.nsr_dir, env=env)

    def _find_strips(self) -> List[str]:
        """Prefere *_nobg.png; senao PNGs com '_cond_'."""
        names = sorted(os.listdir(self._layout.td_output_dir))
        strips = [f for f in names if f.endswith("_nobg.png")] or [
            f for f in names if f.endswith(".png") and "_cond_" in f
        ]
        if not strips:
            raise PipelineError("nenhum strip encontrado em teethdreamer_output")
        return strips
