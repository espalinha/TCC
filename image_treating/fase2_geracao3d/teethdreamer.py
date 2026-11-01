import os
import sys

from commands import CommandRunner
from errors import PipelineError
from pipeline_config import PathLayout, PipelineConfig


class TeethDreamerStep:
    name = "Geracao 2D multi-vista (TeethDreamer)"

    def __init__(self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner) -> None:
        self._config, self._layout, self._runner = config, layout, runner

    def run(self) -> None:
        c, l = self._config, self._layout
        self._prepare_staging()
        cmd = [
            sys.executable, l.td_script,
            "-b", l.td_config,
            "--gpus", "0",
            "--test", l.td_ckpt,
            "--output", l.td_output_dir,
            f"data.params.test_dir={l.staging_dir}",
            "data.params.batch_size=1",
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=c.gpu_id)
        self._runner.run(cmd, cwd=c.teethdreamer_dir, env=env)

    def _prepare_staging(self) -> None:
        """Cria symlinks <uid>_upper / <uid>_lower em staging_dir."""
        out, staging = self._layout.segmentation_dir, self._layout.staging_dir
        if not os.path.isdir(out):
            raise PipelineError(f"saida da fase 1 nao encontrada: {out} (use --segmentation-dir)")
        os.makedirs(staging, exist_ok=True)
        for name in os.listdir(out):
            src, link = os.path.join(out, name), os.path.join(staging, name)
            if name.endswith(("_upper", "_lower")) and os.path.isdir(src) and not os.path.exists(link):
                os.symlink(src, link)
