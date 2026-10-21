"""Etapa 1: segmentacao YOLO (chama segment_teeths.py)."""

from commands import CommandRunner
from pipeline_config import PathLayout, PipelineConfig


class SegmentationStep:
    name = "Segmentacao YOLOv8-seg"

    def __init__(self, config: PipelineConfig, layout: PathLayout, runner: CommandRunner) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError
