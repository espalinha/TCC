import cv2
import numpy as np

MIN_MASKS = 4        # com menos máscaras não há "cor típica de dente" para comparar
RED_K = 3.5          # desvio robusto (MAD) tolerado no a* (vermelho)
RED_FLOOR = 8.0      # desvio mínimo no a* para considerar vermelho demais
LOW_CONF = 0.6       # coroa de ouro também é avermelhada, mas vem com confiança alta


def red_tissue_flags(masks, confs, img_bgr):
    """
    True para cada máscara que é tecido: a cor mediana (Lab) é mais VERMELHA (a*) que a dos outros
    dentes da mesma foto. Só o lado vermelho conta (amálgama e restauração escura também fogem da
    mediana, mas são dente). Coroa de ouro também fica +8..9 em a*, mas o YOLO tem confiança alta nela;
    tecido vem com confiança baixa. Tecido = desvio > lim com conf < LOW_CONF, ou desvio > 2 lim.
    """
    if len(masks) < MIN_MASKS:
        return [False] * len(masks)
    a_star = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    med = np.array([np.median(a_star[m > 0]) for m in masks])
    center = np.median(med)
    lim = max(RED_FLOOR, RED_K * 1.4826 * np.median(np.abs(med - center)))
    dev = med - center
    return [bool((d > lim and c < LOW_CONF) or d > 2 * lim) for d, c in zip(dev, confs)]
