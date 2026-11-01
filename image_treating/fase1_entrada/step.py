import os
import sys

from commands import CommandRunner
from errors import PipelineError
from pipeline_config import HERE, PathLayout, PipelineConfig


class SegmentationStep:
    name = "Segmentacao YOLOv8-seg"

    def __init__(self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner) -> None:
        self._config, self._layout, self._runner = config, layout, runner

    def run(self) -> None:
        c = self._config
        if not c.model_path or not os.path.isfile(c.model_path):
            raise PipelineError(f"{self.name}: pesos do YOLO nao encontrados ({c.model_path}); use --model")
        cmd = [
            sys.executable, "-m", "fase1_entrada.segmentacao",
            "--input-dir", c.input_dir,
            "--output-dir", self._layout.segmentation_dir,
            "--uid", c.uid,
            "--model", c.model_path,
            "--imgsz", str(c.imgsz),
            "--conf", str(c.conf),
        ]
        if c.seg_device:
            cmd += ["--device", c.seg_device]
        if c.debug:
            cmd.append("--debug")
        self._runner.run(cmd, cwd=HERE)
