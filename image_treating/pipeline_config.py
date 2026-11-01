import argparse
import os
from dataclasses import dataclass
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))


def _abspath_or_none(path: Optional[str]) -> Optional[str]:
    return os.path.abspath(path) if path else None


@dataclass(frozen=True)
class PipelineConfig:
    """Parametros imutaveis do pipeline. Nenhum caminho depende da posicao deste arquivo."""

    input_dir: str
    output_dir: str
    uid: str
    teethdreamer_dir: str  # injetado: antes era THIS_DIR/../TeethDreamer
    model_path: Optional[
        str
    ]  # pesos do YOLO; fora do repositorio (ex.: /scratch/.../checkpoints/yolo/best.pt)
    imgsz: int = 1920
    conf: float = 0.18
    seg_device: str = "cpu"
    debug: bool = False
    gpu_id: str = "0"
    cpu: int = 4
    neus_config: Optional[str] = None
    refiner_dir: Optional[str] = (
        None  # pasta scripts/ com posprocessar.py e oclusal/esculpir_neus.py
    )
    refiner_python: str = (
        "python"  # python do env do refinador (ex.: conda env teeth_refiner)
    )
    sculpt_checkpoint: Optional[str] = None
    sculpt_alpha: float = 1.0
    gt_dir: Optional[str] = (
        None  # pasta com o STL real de cada arcada; liga as metricas (fase 3)
    )
    raw_dir: Optional[str] = None  # fase 0: acervo bruto (somente leitura)
    organize_overwrite: bool = False
    # pastas de cada fase (a saida de uma e a entrada da seguinte); None = subpasta de output_dir
    organized_dir: Optional[str] = None  # fase 0 -> treated/ e with_stl/
    segmentation_dir: Optional[str] = (
        None  # fase 1 -> <uid>_upper/ e <uid>_lower/ (entrada da fase 2)
    )
    generation_dir: Optional[str] = (
        None  # fase 2 -> teethdreamer_output/ e reconstructions/ (entrada da fase 3)
    )
    results_dir: Optional[str] = (
        None  # fase 3 -> postprocessed/, sculpted/, metricas.json e final/
    )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PipelineConfig":
        input_dir = os.path.abspath(args.input_dir)
        return cls(
            input_dir=input_dir,
            output_dir=os.path.abspath(args.output_dir),
            uid=args.uid or os.path.basename(input_dir.rstrip(os.sep)),
            teethdreamer_dir=os.path.abspath(args.teethdreamer_dir),
            model_path=_abspath_or_none(args.model),
            imgsz=args.imgsz,
            conf=args.conf,
            seg_device=args.seg_device,
            debug=args.debug,
            gpu_id=args.gpu_id,
            cpu=args.cpu,
            neus_config=args.neus_config,
            refiner_dir=_abspath_or_none(args.refiner_dir),
            refiner_python=args.refiner_python,
            sculpt_checkpoint=_abspath_or_none(args.sculpt_checkpoint),
            sculpt_alpha=args.sculpt_alpha,
            gt_dir=_abspath_or_none(args.gt),
            raw_dir=_abspath_or_none(args.raw_dir),
            organize_overwrite=args.organize_overwrite,
            organized_dir=_abspath_or_none(args.organized_dir),
            segmentation_dir=_abspath_or_none(args.segmentation_dir),
            generation_dir=_abspath_or_none(args.generation_dir),
            results_dir=_abspath_or_none(args.results_dir),
        )


class PathLayout:
    """Caminhos derivados de um PipelineConfig (entrada, saidas e ferramentas externas)."""

    def __init__(self, config: PipelineConfig) -> None:
        out, td = config.output_dir, config.teethdreamer_dir
        self.organized_dir = config.organized_dir or os.path.join(
            out, "fase0_organizacao"
        )
        self.treated_dir = os.path.join(self.organized_dir, "treated")
        self.with_stl_dir = os.path.join(self.organized_dir, "with_stl")
        self.segmentation_dir = config.segmentation_dir or os.path.join(
            out, "fase1_entrada"
        )
        gen = config.generation_dir or os.path.join(out, "fase2_geracao3d")
        self.generation_dir = gen
        self.staging_dir = os.path.join(gen, ".td_staging")
        self.td_output_dir = os.path.join(gen, "teethdreamer_output")
        self.recon_dir = os.path.join(gen, "reconstructions")
        res = config.results_dir or os.path.join(out, "fase3_posprocessamento")
        self.results_dir = res
        self.post_dir = os.path.join(res, "postprocessed")
        self.sculpt_dir = os.path.join(res, "sculpted")
        self.metrics_json = os.path.join(res, "metricas.json")
        self.final_dir = os.path.join(res, "final")
        self.final_images_dir = os.path.join(self.final_dir, "imagens")
        self.nsr_dir = os.path.join(td, "instant-nsr-pl")
        self.td_script = os.path.join(td, "TeethDreamer.py")
        self.td_config = os.path.join(td, "configs", "TeethDreamer.yaml")
        self.td_ckpt = os.path.join(td, "ckpt", "TeethDreamer.ckpt")
        refiner = config.refiner_dir or ""
        self.post_script = os.path.join(refiner, "posprocessar.py")
        self.sculpt_script = os.path.join(refiner, "oclusal", "esculpir_neus.py")
