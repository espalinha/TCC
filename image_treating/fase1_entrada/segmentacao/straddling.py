import cv2
import numpy as np

from fase1_entrada.segmentacao.masks import fill_holes

STRADDLE_FRAC = 0.30  # fração de massa no lado oposto da fronteira -> máscara U+L fundida


def _valley_cut(proj, default_thr):
    """
    Acha o 'vale' entre dois lobos no perfil 1D da projeção da máscara
    (padrão bimodal de superior+inferior fundidos). Se o perfil não for
    claramente bimodal, retorna a fronteira padrão.
    """
    hist, edges = np.histogram(proj, bins=48)
    if hist.sum() == 0:
        return default_thr
    h = cv2.GaussianBlur(hist.astype(np.float32)[None, :], (1, 9), 2.0).flatten()
    if h.max() <= 0:
        return default_thr

    # dois picos mais altos com separação mínima
    order = np.argsort(h)[::-1]
    p1 = int(order[0])
    p2 = -1
    min_sep = max(4, len(h) // 4)
    for idx in order[1:]:
        if abs(int(idx) - p1) >= min_sep and h[idx] > 0.25 * h[p1]:
            p2 = int(idx)
            break
    if p2 < 0:
        return default_thr

    lo, hi = sorted([p1, p2])
    valley = lo + int(np.argmin(h[lo:hi + 1]))
    if h[valley] > 0.6 * min(h[lo], h[hi]):  # vale real: bem mais baixo que os dois picos
        return default_thr
    return float(edges[valley] + edges[valley + 1]) * 0.5


def separate_straddling(upper_masks, lower_masks, boundary, min_area):
    """
    Separa máscaras que cruzam a fronteira entre as arcadas (dente superior +
    inferior fundidos pelo modelo num blob só):
      - máscara com massa quase toda de um lado -> atribui pelo centroide;
      - máscara com massa relevante dos dois lados -> corta no vale do perfil
        (ou na fronteira) e cada pedaço vai para a sua arcada.
    Regra única e geométrica: nenhum dente é descartado.
    Retorna (upper, lower, n_separadas).
    """
    if boundary is None:
        return upper_masks, lower_masks, 0
    mean, v, thr = boundary

    new_u, new_l = [], []
    n_cut = 0
    for m in list(upper_masks) + list(lower_masks):
        ys, xs = np.where(m > 0)
        if len(ys) == 0:
            continue
        proj = (xs - mean[0]) * v[0] + (ys - mean[1]) * v[1]
        below_frac = float((proj > thr).mean())
        whole_side = new_l if float(np.median(proj)) > thr else new_u

        if below_frac <= STRADDLE_FRAC or below_frac >= 1.0 - STRADDLE_FRAC:
            whole_side.append(m)
            continue

        sel = proj <= _valley_cut(proj, thr)
        top, bot = np.zeros_like(m), np.zeros_like(m)
        top[ys[sel], xs[sel]] = 1
        bot[ys[~sel], xs[~sel]] = 1
        top, bot = fill_holes(top), fill_holes(bot)
        if int(top.sum()) >= min_area and int(bot.sum()) >= min_area:
            new_u.append(top)
            new_l.append(bot)
            n_cut += 1
        else:
            whole_side.append(m)

    if n_cut:
        print(f"  {n_cut} máscara(s) U+L fundida(s) separada(s) na fronteira")
    return new_u, new_l, n_cut
