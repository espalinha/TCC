from dataclasses import replace

import cv2
import numpy as np

from fase1_entrada.segmentacao import occlusal
from fase1_entrada.segmentacao.arch_split import split_upper_lower
from fase1_entrada.segmentacao.shared_teeth import resolve_shared_teeth
from fase1_entrada.segmentacao.straddling import separate_straddling
from fase1_entrada.segmentacao.color import red_tissue_flags
from fase1_entrada.segmentacao.confidence import adaptive_conf_threshold
from fase1_entrada.segmentacao.masks import absorb_defects, clean_mask, complete_convex, remove_outlier_masks
from fase1_entrada.segmentacao.views import VIEW_UPPER_OCC

MIN_AREA_FRAC = 0.0002  # área mínima de um dente (fração da imagem) -> remove blobs falsos
WORK_MAX_SIDE = 1500    # resolução de trabalho (lado maior); a saída do TeethDreamer tem 1000 px


def _min_area(img):
    return int(MIN_AREA_FRAC * img.shape[0] * img.shape[1])


def _to_work_size(img):
    h, w = img.shape[:2]
    s = WORK_MAX_SIDE / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def _drop(state, flags, label):
    """Tira de `masks` (e de `mask_confs`) as máscaras marcadas, avisando quantas saíram."""
    keep = [not f for f in flags]
    n = len(keep) - sum(keep)
    if n:
        print(f"  {n} máscara(s) descartada(s): {label}")
    return replace(state, masks=[m for m, k in zip(state.masks, keep) if k],
                   mask_confs=[c for c, k in zip(state.mask_confs, keep) if k])


def detect(detector):
    """Fábrica da etapa de detecção (o detector é a única dependência injetada)."""
    def stage(state):
        img = cv2.imread(state.path)
        if img is None:
            print(f"  [ERRO] não consegui ler {state.path}")
            return replace(state, stopped=True)
        img = _to_work_size(img)
        found = detector.detect(state.path, img.shape)
        if not found.masks:
            print(f"  [AVISO] nenhum dente detectado em {state.name}")
        return replace(state, img=img, raw_masks=found.masks, confs=found.confs, defects=found.defects)
    return stage


def _keep_confident(state, thr):
    keep = [float(c) >= thr for c in state.confs]
    return replace(state, raw_masks=[m for m, k in zip(state.raw_masks, keep) if k],
                   confs=state.confs[np.array(keep, bool)])


def filter_by_base_confidence(state):
    """Laterais/frontal: corte fixo no --conf."""
    return _keep_confident(state, state.conf)


def filter_by_confidence(state):
    """Oclusais: descarta falsos positivos (pele, luva, fundo) pelo gap de confiança (nunca abaixo do --conf)."""
    if not state.raw_masks:
        return state
    thr = adaptive_conf_threshold(state.confs, state.conf)
    if thr > state.conf:
        print(f"  limiar de confiança adaptativo: {thr:.2f} (k-means no gap das confianças)")
    return _keep_confident(state, thr)


def clean_masks(state):
    min_area = _min_area(state.img)
    cleaned = [(clean_mask(m, min_area), float(c)) for m, c in zip(state.raw_masks, state.confs)]
    return replace(state, masks=[m for m, _ in cleaned if m is not None],
                   mask_confs=[c for m, c in cleaned if m is not None])


def drop_red_tissue(state):
    """Lábio/gengiva/língua marcados como dente (mais vermelhos que os outros dentes da foto)."""
    return _drop(state, red_tissue_flags(state.masks, state.mask_confs, state.img), "tecido (cor)")


def drop_outliers(state):
    masks, n = remove_outlier_masks(state.masks)
    if n:
        print(f"  {n} detecção(ões) isolada(s) descartada(s) (lixo longe do grupo de dentes)")
    kept = {id(m) for m in masks}
    return replace(state, masks=masks, n_outliers=n,
                   mask_confs=[c for m, c in zip(state.masks, state.mask_confs) if id(m) in kept])


def drop_tissue_shapes(state):
    return _drop(state, occlusal.tissue_shape_flags(state.masks, state.mask_confs), "forma de tecido")


def drop_border_blobs(state):
    return _drop(state, occlusal.border_blob_flags(state.masks), "lábio/dedo na borda")


def drop_opposite_arch(state):
    flags = occlusal.opposite_arch_flags(state.masks, upper=state.view == VIEW_UPPER_OCC)
    return _drop(state, flags, "arcada oposta")


def drop_off_arch(state):
    return _drop(state, occlusal.off_arch_flags(state.masks), "fora da curva da arcada")


def absorb_defect_masks(state):
    """Cárie/cavidade/trinca sobre um dente vira parte dele (senão fica buraco preto)."""
    return replace(state, masks=absorb_defects(state.masks, state.defects))


def complete_teeth(state):
    """Completa entalhes pequenos de cada dente com o casco convexo."""
    return replace(state, masks=[complete_convex(m) for m in state.masks])


def split_arches(state):
    upper, lower, boundary = split_upper_lower(state.masks, state.img.shape[0])
    return replace(state, upper=upper, lower=lower, boundary=boundary)


def resolve_shared(state):
    upper, lower, n = resolve_shared_teeth(state.upper, state.lower)
    if n:
        print(f"  {n} dente(s) em duas arcadas resolvido(s) pela vizinhança "
              f"(dente acima -> inferior, dente abaixo -> superior)")
    return replace(state, upper=upper, lower=lower)


def separate_fused(state):
    upper, lower, _ = separate_straddling(state.upper, state.lower, state.boundary, _min_area(state.img))
    return replace(state, upper=upper, lower=lower)


def whole_arch(state):
    """Oclusais: arcada única, nunca dividida. Sem dentes -> o workflow para e nada é gravado."""
    if not state.masks:
        if state.raw_masks:
            print(f"  [AVISO] máscara vazia em {state.name}")
        return replace(state, stopped=True)
    if state.view == VIEW_UPPER_OCC:
        return replace(state, upper=state.masks)
    return replace(state, lower=state.masks)
