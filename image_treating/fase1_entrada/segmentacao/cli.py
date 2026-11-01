import argparse
import os
import sys

from fase1_entrada.segmentacao import workflow
from fase1_entrada.segmentacao.config import SegmentationConfig
from fase1_entrada.segmentacao.detector import YoloDetector, load_yolo
from fase1_entrada.segmentacao.outputs import summary, write_debug, write_report, write_teethdreamer
from fase1_entrada.segmentacao.state import ViewState
from fase1_entrada.segmentacao.views import REQUIRED_VIEWS, VIEW_LOWER_OCC, VIEW_UPPER_OCC, find_views, is_named_occlusal

LATERAL_OUTPUTS = [write_teethdreamer, write_debug]
OCCLUSAL_OUTPUTS = [write_teethdreamer]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Segmenta dentes das 5 fotos intraorais e gera pastas prontas para o TeethDreamer."
    )
    parser.add_argument("--input-dir", default="test_images", help="Pasta com as fotos de UM paciente: 3 laterais/frontal e, opcionalmente, 2 oclusais (padrão: test_images)")
    parser.add_argument("--output-dir", default="teethdreamer_out", help="Pasta de saída (padrão: teethdreamer_out)")
    parser.add_argument("--uid", default=None, help="Nome do caso (padrão: nome da pasta de entrada)")
    parser.add_argument("--model", default="best.pt", help="Pesos do modelo (padrão: best.pt)")
    parser.add_argument("--imgsz", type=int, default=1920, help="Resolução de inferência (padrão: 1920)")
    parser.add_argument("--conf", type=float, default=0.10, help="Confiança mínima (padrão: 0.10)")
    parser.add_argument("--device", default=None, help="Device de inferência (ex: 0, cpu). Padrão: automático (GPU se houver)")
    parser.add_argument("--debug", action="store_true", help="Salva overlays e recortes RGBA em <output>/debug/")
    return parser


def _process(path, view, config, stages, outputs):
    print(f"\n[{view}] {os.path.basename(path)}")
    state = workflow.run(ViewState(view, path, conf=config.conf), stages)
    if state.stopped:
        return None
    print(f"  {summary(state)}")
    for output in outputs:
        output(state, config)
    return state


def _find_required_views(config):
    """Vistas laterais/frontal do paciente; encerra com erro se faltar alguma."""
    print(f"Caso: {config.uid}")
    print(f"Procurando as views {REQUIRED_VIEWS} em: {config.input_dir}")
    views = find_views(config.input_dir)
    missing = [v for v in REQUIRED_VIEWS if v not in views]
    if missing:
        print(f"\n[ERRO] views faltando: {missing}")
        print(f"Esperado: {REQUIRED_VIEWS}")
        sys.exit(1)
    return views


def _process_occlusals(views, config, detector):
    """Oclusais: só arquivos "ocl*" (as numeradas 3.jpg/4.jpg continuam ignoradas, como antes)."""
    for view in (VIEW_UPPER_OCC, VIEW_LOWER_OCC):
        path = views.get(view)
        if path and is_named_occlusal(path):
            _process(path, view, config, workflow.occlusal(detector), OCCLUSAL_OUTPUTS)


def main(argv=None):
    config = SegmentationConfig.from_args(build_parser().parse_args(argv))
    views = _find_required_views(config)

    print(f"Carregando modelo: {config.model}")
    detector = YoloDetector(load_yolo(config.model), config.imgsz, config.conf, config.device)

    states = [_process(views[v], v, config, workflow.lateral(detector), LATERAL_OUTPUTS) for v in REQUIRED_VIEWS]
    states = [s for s in states if s is not None]
    _process_occlusals(views, config, detector)

    os.makedirs(config.output_dir, exist_ok=True)
    report = write_report(states, config)
    print(f"\nPronto! Estrutura TeethDreamer em: {config.output_dir}")
    print(f"  {config.uid}_upper/000..003.png   (front, left, right, oclusal superior)")
    print(f"  {config.uid}_lower/000..003.png   (front, left, right, oclusal inferior)")
    print(f"Relatório: {report}")
