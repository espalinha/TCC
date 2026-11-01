import cv2
import numpy as np

OUTLIER_NN_FACTOR = (
    3.0  # descarta máscara se vizinho mais próximo > 3x a distância mediana do grupo
)
MIN_OUTLIER_CHECK = 5  # com menos máscaras que isso não há grupo para comparar
TOOTH_CLOSE_FRAC = 0.06  # fecha fissuras de até ~6% do tamanho do dente (raiz da área)
HULL_MIN_SOLIDITY = (
    0.80  # entalhe pequeno (solidez acima disso): completa o dente com o casco convexo
)
ARCH_GAP_FRAC = (
    0.18  # fecha vãos entre dentes vizinhos de até ~18% do tamanho mediano do dente
)
ARCH_MIN_PIECE = 0.02  # pedaço solto < 2% do maior pedaço da arcada é sujeira
DEFECT_TOUCH = (
    7  # defeito (cárie/restauração) a até 7 px de um dente conta como sobre o dente
)


def _ellipse(size):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _odd(x):
    return max(3, int(x) | 1)


def fill_holes(mask):
    """Preenche buracos internos da máscara: o fundo ligado à borda é apagado por flood-fill;
    o que sobra do complemento é buraco."""
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return m
    h, w = m.shape
    inv = np.pad(1 - m, 1, constant_values=1)
    cv2.floodFill(inv, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
    return np.maximum(m, inv[1:-1, 1:-1])


def largest_component(mask):
    """Só o maior componente conexo (8-vizinhança); zeros se a máscara estiver vazia."""
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m)
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def clean_mask(mask, min_area):
    """Binariza (> 0.5), mantém o maior componente, fecha fissuras na escala do próprio dente
    e preenche buracos. None se ficar menor que `min_area`."""
    m = largest_component(mask > 0.5)
    if int(m.sum()) < max(1, min_area):
        return None
    m = cv2.morphologyEx(
        m, cv2.MORPH_CLOSE, _ellipse(_odd(TOOTH_CLOSE_FRAC * np.sqrt(m.sum())))
    )
    return fill_holes(largest_component(m))


def solidity(mask):
    """(área / área do casco convexo, casco) do maior contorno; (0, None) se vazia."""
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0, None
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    return (cv2.contourArea(c) / hull_area if hull_area > 0 else 1.0), hull


def complete_convex(mask):
    """Dente com entalhe (canto cortado pela caixa do YOLO, ponta comida): na foto a coroa é ~convexa,
    então um entalhe pequeno é completado pelo casco convexo. Forma muito côncava fica como está.
    """
    s, hull = solidity(mask)
    if hull is None or s >= 0.995 or s < HULL_MIN_SOLIDITY:
        return mask
    out = np.zeros_like(mask)
    cv2.drawContours(out, [hull], -1, 1, cv2.FILLED)
    return out


def absorb_defects(teeth, defects):
    """Cárie, cavidade e trinca são classes separadas do YOLO; sobre um dente, viram parte dele
    (senão a restauração/cárie fica como buraco preto). Vão para o dente com maior sobreposição.
    """
    out = [t.copy() for t in teeth]
    if not out:
        return out
    near = np.ones((DEFECT_TOUCH, DEFECT_TOUCH), np.uint8)
    for d in defects:
        d = (d > 0).astype(np.uint8)
        if d.sum() == 0:
            continue
        overlap = [int((t & d).sum()) for t in out]
        if max(overlap) == 0:
            grown = cv2.dilate(d, near)
            overlap = [int((t & grown).sum()) for t in out]
        if max(overlap) > 0:
            i = int(np.argmax(overlap))
            out[i] = fill_holes(out[i] | d)
    return out


def arch_mask(masks, shape):
    """Máscara final da arcada: união dos dentes, vãos entre vizinhos fechados com kernel proporcional
    ao tamanho mediano do dente (não da imagem), buracos preenchidos e pedaços soltos minúsculos removidos.
    """
    out = np.zeros(shape[:2], np.uint8)
    if not masks:
        return out
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    size = np.sqrt(np.median([m.sum() for m in masks]))
    out = fill_holes(
        cv2.morphologyEx(out, cv2.MORPH_CLOSE, _ellipse(_odd(ARCH_GAP_FRAC * size)))
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    if n > 2:
        big = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < ARCH_MIN_PIECE * big:
                out[labels == i] = 0
    return out


def combine_masks(mask_list, shape):
    """União (máximo pixel a pixel) das máscaras, no tamanho (altura, largura) de `shape`."""
    combined = np.zeros(shape[:2], dtype=np.uint8)
    for m in mask_list:
        combined = np.maximum(combined, m)
    return combined


def bbox(mask):
    """(x_min, y_min, x_max, y_max, cx, cy) da máscara; ValueError se estiver vazia."""
    ys, xs = np.where(mask > 0)
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
        float(xs.mean()),
        float(ys.mean()),
    )


def touches_border(mask, margin=3):
    x0, y0, x1, y1 = bbox(mask)[:4]
    h, w = mask.shape[:2]
    return x0 <= margin or y0 <= margin or x1 >= w - 1 - margin or y1 >= h - 1 - margin


def _nearest_neighbor_distances(masks):
    """Distância de cada centroide ao centroide mais próximo."""
    cents = np.float32([bbox(m)[4:] for m in masks])
    dist = np.sqrt(((cents[:, None, :] - cents[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return dist.min(axis=1)


def remove_outlier_masks(masks):
    """
    Remove detecções isoladas (lixo longe do grupo de dentes, ex.: blob na luva
    ou no canto da imagem): se a distância do centroide da máscara ao vizinho
    mais próximo for muito maior que a mediana do grupo, é outlier.
    Retorna (máscaras mantidas, nº descartadas).
    """
    if len(masks) < MIN_OUTLIER_CHECK:
        return masks, 0
    d_nearest = _nearest_neighbor_distances(masks)
    keep = d_nearest <= np.median(d_nearest) * OUTLIER_NN_FACTOR
    kept = [m for m, k in zip(masks, keep) if k]
    return kept, len(masks) - len(kept)
