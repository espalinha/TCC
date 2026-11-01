import cv2
import numpy as np

SPLIT_MIN_SEP_TOOTH = (
    0.5  # separação mínima entre arcadas (fração da altura mediana do dente)
)
MIN_BALANCE_CHECK = (
    6  # a partir daqui, um cluster com <15% dos dentes é tratado como outlier
)
MIN_CLUSTER_FRAC = 0.15
AXIS_TILT_MIN = 0.3  # |v_y| menor que isso: eixo quase horizontal, usa o Y da imagem


def _single_arch(masks, mean_y, img_h):
    """Só uma arcada visível: decide qual pela posição vertical média."""
    return (masks, []) if mean_y < img_h / 2 else ([], masks)


def _arch_axis(pts):
    """Eixo perpendicular à arcada = direção de MENOR variância (PCA).
    Retorna (centro, eixo, projeção dos pontos no eixo, como coluna float32)."""
    mean = pts.mean(axis=0)
    _, eigvecs = np.linalg.eigh(
        np.cov((pts - mean).T)
    )  # autovetores em ordem crescente
    v = eigvecs[:, 0].astype(np.float64)
    if abs(v[1]) < AXIS_TILT_MIN:
        v = np.array([0.0, 1.0])
        proj = pts[:, 1] - mean[1]
    else:
        if v[1] < 0:
            v = -v  # orienta: projeção maior = mais abaixo na imagem
        proj = (pts - mean) @ v
    return mean, v, proj.astype(np.float32).reshape(-1, 1)


def _kmeans_two_groups(proj):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(proj, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return labels.flatten(), centers.flatten()


def split_upper_lower(masks, img_h):
    """
    Divide os dentes em superior/inferior:
    1) projeta os centroides no eixo perpendicular à arcada (PCA) — funciona
       mesmo com o plano oclusal inclinado nas fotos laterais;
    2) k-means (k=2) na projeção;
    3) checagens de sanidade: separação mínima (arcada única) e clusters
       desbalanceados (k-means isolou outlier -> refaz corte na mediana).
    Retorna (upper_masks, lower_masks, boundary), onde boundary = (mean, v, thr)
    descreve a fronteira entre as arcadas no eixo da projeção (None se arcada única).
    """
    entries = []  # (cx, cy, altura_bbox, máscara)
    for m in masks:
        ys, xs = np.where(m > 0)
        if len(ys) > 0:
            entries.append(
                (float(xs.mean()), float(ys.mean()), float(ys.max() - ys.min()), m)
            )

    if not entries:
        return [], [], None
    if len(entries) == 1:
        upper, lower = _single_arch([entries[0][3]], entries[0][1], img_h)
        return upper, lower, None

    pts = np.float32([[e[0], e[1]] for e in entries])
    med_h = float(np.median([e[2] for e in entries]))
    mean, v, proj = _arch_axis(pts)
    labels, centers = _kmeans_two_groups(proj)

    if abs(centers[0] - centers[1]) < med_h * SPLIT_MIN_SEP_TOOTH:  # arcada única
        upper, lower = _single_arch(
            [e[3] for e in entries], float(pts[:, 1].mean()), img_h
        )
        return upper, lower, None

    counts = np.bincount(labels, minlength=2)
    if len(entries) >= MIN_BALANCE_CHECK and counts.min() < max(
        2, int(MIN_CLUSTER_FRAC * len(entries))
    ):
        # k-means isolou um outlier: refaz o corte na mediana (sempre balanceado)
        thr = float(np.median(proj))
        labels = (proj.flatten() >= thr).astype(np.int32)
        upper_label = 0
    else:
        thr = float(0.5 * (centers[0] + centers[1]))  # fronteira = meio dos clusters
        upper_label = int(np.argmin(centers))
    upper = [e[3] for e, lb in zip(entries, labels) if lb == upper_label]
    lower = [e[3] for e, lb in zip(entries, labels) if lb != upper_label]
    return upper, lower, (mean, v, thr)
