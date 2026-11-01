import cv2
import numpy as np

MIN_DETECTIONS = 4  # com menos que isso não há dois grupos para separar
MIN_GAP = 0.15  # diferença mínima entre os grupos (e entre mínimo e máximo)
MEDIAN_CAP = 0.85  # o limiar nunca passa de 85% da mediana: não corta o grupo principal


def adaptive_conf_threshold(confs, base_conf):
    """
    Limiar de confiança adaptativo: k-means 1D (k=2) nas confianças. Se elas
    formam dois grupos separados (dentes reais com conf alta vs. falsos
    positivos com conf baixa — típico em oclusais: pele, luva, fundo),
    retorna o ponto médio entre os centros dos grupos. Caso contrário
    (confianças homogêneas ou grupo baixo majoritário), retorna o conf base.
    """
    if len(confs) < MIN_DETECTIONS:
        return base_conf
    if float(confs.max()) - float(confs.min()) < MIN_GAP:
        return base_conf  # grupo homogêneo: nada a separar

    x = confs.astype(np.float32).reshape(-1, 1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, labels, centers = cv2.kmeans(x, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.flatten()
    if abs(centers[1] - centers[0]) < MIN_GAP:
        return base_conf

    counts = np.bincount(labels.flatten(), minlength=2)
    low_cluster = int(np.argmin(centers))
    if counts[low_cluster] >= counts[1 - low_cluster]:
        return base_conf  # grupo baixo não é exceção minoritária

    thr = min(
        float(0.5 * (centers[0] + centers[1])), MEDIAN_CAP * float(np.median(confs))
    )
    return max(base_conf, thr)
