import cv2
import numpy as np

OUTPUT_SIZE = (
    1000  # tamanho das imagens finais (quadradas), igual ao example do TeethDreamer
)

UPPER_COLOR = (0, 200, 0)  # BGR: verde
LOWER_COLOR = (255, 120, 0)  # BGR: azul
CENTROID_COLOR = (0, 0, 255)
LEGEND = "verde=superior  azul=inferior"


def _pad_to_square(img, pad):
    """Borda centralizada que torna a imagem quadrada; `pad` é o valor da borda (0 ou (0, 0, 0))."""
    h, w = img.shape[:2]
    side = max(h, w)
    top, left = (side - h) // 2, (side - w) // 2
    return cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left, cv2.BORDER_CONSTANT, value=pad
    )


def apply_mask_square(img_bgr, mask, out_size=OUTPUT_SIZE):
    """Padding preto até quadrado, aplica a máscara (fundo preto) e redimensiona."""
    square = _pad_to_square(img_bgr, (0, 0, 0))
    sq_mask = _pad_to_square(mask, 0)
    out = cv2.bitwise_and(square, square, mask=sq_mask)
    return cv2.resize(out, (out_size, out_size), interpolation=cv2.INTER_AREA)


def mask_to_rgba(img_bgr, mask):
    """Foto com a máscara como canal alfa (recorte com fundo transparente)."""
    b, g, r = cv2.split(img_bgr)
    return cv2.merge([b, g, r, (mask * 255).astype(np.uint8)])


def _paint_masks(overlay, masks, color):
    """Pinta cada máscara (preenchimento translúcido, contorno e centroide) sobre `overlay`."""
    for m in masks:
        colored = np.zeros_like(overlay)
        colored[m > 0] = color
        overlay[:] = cv2.addWeighted(colored, 0.45, overlay, 1.0, 0)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 6)
        ys, xs = np.where(m > 0)
        cv2.circle(overlay, (int(xs.mean()), int(ys.mean())), 14, CENTROID_COLOR, -1)


def draw_overlay(img_bgr, upper_masks, lower_masks, max_dim=1600):
    """Overlay para inspeção: verde = superior, azul = inferior."""
    overlay = img_bgr.copy()
    _paint_masks(overlay, upper_masks, UPPER_COLOR)
    _paint_masks(overlay, lower_masks, LOWER_COLOR)

    h, w = overlay.shape[:2]
    if max(h, w) > max_dim:
        s = max_dim / max(h, w)
        overlay = cv2.resize(overlay, (int(w * s), int(h * s)))
    cv2.putText(
        overlay, LEGEND, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
    )
    return overlay
