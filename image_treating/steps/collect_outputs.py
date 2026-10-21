"""Etapa 5: coleta .obj finais (e imagens geradas) em final/."""

from typing import Dict, Optional

from pipeline_config import PathLayout, PipelineConfig


class CollectOutputsStep:
    name = "Coleta de saidas finais"

    def __init__(self, config: PipelineConfig, layout: PathLayout) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError

    def _copy_images(self) -> int:
        raise NotImplementedError

    def _collect_arch(self, arch: str) -> Optional[str]:
        raise NotImplementedError

    @staticmethod
    def _find_final_obj(arch_dir: str) -> Optional[str]:
        """.obj de maior iteracao em <arch_dir>/*/neus/save/*.obj."""
        raise NotImplementedError

    @staticmethod
    def _export_stl(obj_path: str) -> None:
        raise NotImplementedError
