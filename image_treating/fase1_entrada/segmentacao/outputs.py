import os

import cv2

from fase1_entrada.segmentacao.masks import arch_mask
from fase1_entrada.segmentacao.render import apply_mask_square, draw_overlay, mask_to_rgba
from fase1_entrada.segmentacao.views import VIEW_FRONT, VIEW_LEFT, VIEW_LOWER_OCC, VIEW_RIGHT, VIEW_UPPER_OCC

# índice do arquivo dentro de cada pasta do TeethDreamer
UPPER_INDEX = {VIEW_FRONT: 0, VIEW_LEFT: 1, VIEW_RIGHT: 2, VIEW_UPPER_OCC: 3}
LOWER_INDEX = {VIEW_FRONT: 0, VIEW_LEFT: 1, VIEW_RIGHT: 2, VIEW_LOWER_OCC: 3}


def _arch_mask(masks, img):
    return arch_mask(masks, img.shape)


def write_teethdreamer(state, config):
    """RGB quadrado, fundo preto, só os dentes da arcada: <uid>_upper|lower/NNN.png."""
    for arch, index, masks in (("upper", UPPER_INDEX, state.upper), ("lower", LOWER_INDEX, state.lower)):
        if state.view in index:
            folder = os.path.join(config.output_dir, f"{config.uid}_{arch}")
            os.makedirs(folder, exist_ok=True)
            out = apply_mask_square(state.img, _arch_mask(masks, state.img))
            cv2.imwrite(os.path.join(folder, f"{index[state.view]:03d}.png"), out)


def write_debug(state, config):
    """Overlay colorido e recorte RGBA em <output>/debug/ (só com --debug)."""
    if not config.debug:
        return
    folder = os.path.join(config.output_dir, "debug")
    os.makedirs(folder, exist_ok=True)
    base = f"{state.view}_{os.path.splitext(state.name)[0]}"
    overlay = draw_overlay(state.img, state.upper, state.lower)
    cv2.imwrite(os.path.join(folder, f"{base}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])
    rgba = mask_to_rgba(state.img, _arch_mask(state.upper + state.lower, state.img))
    h, w = rgba.shape[:2]
    if max(h, w) > 2000:
        s = 2000 / max(h, w)
        rgba = cv2.resize(rgba, (int(w * s), int(h * s)))
    cv2.imwrite(os.path.join(folder, f"{base}_rgba.png"), rgba)


def summary(state):
    n = len(state.upper) + len(state.lower)
    if not n:
        return f"{state.view}: NENHUM DENTE DETECTADO"
    return (f"{state.view}: {n} dentes -> {len(state.upper)} superior / {len(state.lower)} inferior "
            f"(conf média {state.confs.mean():.2f}, {state.n_outliers} outliers descartados)")


def write_report(states, config):
    path = os.path.join(config.output_dir, f"{config.uid}_report.txt")
    with open(path, "w") as f:
        f.write(f"Caso: {config.uid}\nimgsz={config.imgsz} conf={config.conf}\n\n")
        f.write("\n".join(summary(s) for s in states) + "\n")
    return path
