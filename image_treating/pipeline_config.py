"""Configuracao do pipeline: parametros (PipelineConfig) e caminhos derivados (PathLayout)."""

import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Parametros imutaveis do pipeline. Nenhum caminho depende da posicao deste arquivo."""

    input_dir: str
    output_dir: str
    uid: str
    teethdreamer_dir: str  # injetado: antes era THIS_DIR/../TeethDreamer
    model_path: str  # injetado: antes era THIS_DIR/<model>
    imgsz: int = 1920
    conf: float = 0.18
    seg_device: str = "cpu"
    debug: bool = False
    gpu_id: str = "0"
    cpu: int = 4
    neus_config: Optional[str] = None

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        raise NotImplementedError


class PathLayout:
    """Caminhos derivados de um PipelineConfig (entrada, saidas e ferramentas externas)."""

    def __init__(self, config: PipelineConfig) -> None:
        raise NotImplementedError

    @property
    def staging_dir(self) -> str:
        raise NotImplementedError

    @property
    def td_output_dir(self) -> str:
        raise NotImplementedError

    @property
    def recon_dir(self) -> str:
        raise NotImplementedError

    @property
    def final_dir(self) -> str:
        raise NotImplementedError

    @property
    def final_images_dir(self) -> str:
        raise NotImplementedError

    @property
    def nsr_dir(self) -> str:
        raise NotImplementedError
