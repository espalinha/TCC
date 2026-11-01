import numpy as np

from fase1_entrada.segmentacao.masks import bbox, solidity, touches_border

MIN_SOLIDITY = 0.65         # área/casco abaixo disso: tecido enrolado/rasgado, não dente
BORDER_MIN_SOLIDITY = 0.85  # idem, mais rígido para máscara encostada na borda
HIGH_CONF = 0.85            # dente com entalhe grande e confiança alta fica...
HARD_MIN_SOLIDITY = 0.5     # ...a não ser que a solidez seja muito baixa
BORDER_BLOB_FACTOR = 1.5    # na borda, maior que 1.5x o maior dente de dentro = lábio/dedo/afastador
OPEN_CENTER_FRAC = 0.35     # "central": |x - centro| < 35% da meia largura da arcada
OPEN_SIDE_FRAC = 0.45       # "lado aberto": 45% da altura da arcada no lado oposto aos incisivos
OPEN_SMALL_FRAC = 0.4       # pedaço na borda do lado aberto menor que 40% da área mediana
CURVE_MIN_MASKS = 6         # com menos máscaras a parábola não é confiável
CURVE_MAX_DIST = 2.0        # distância máxima à curva, em "tamanhos de dente" (raiz da área mediana)
CURVE_MIN_KEEP = 0.6        # se a curva descartaria mais de 40%, o ajuste é que está ruim: não descarta


def _centroids(masks):
    return np.array([bbox(m)[4:] for m in masks], float)


def tissue_shape_flags(masks, confs):
    """Forma de tecido (solidez baixa). Rodar ANTES de completar com o casco convexo."""
    flags = []
    for m, c in zip(masks, confs):
        s, _ = solidity(m)
        lim = BORDER_MIN_SOLIDITY if touches_border(m) else MIN_SOLIDITY
        flags.append(not (s >= lim or (c >= HIGH_CONF and s >= HARD_MIN_SOLIDITY)))
    return flags


def border_blob_flags(masks):
    """Lábio/dedo/afastador na borda: encosta na borda e é maior que BORDER_BLOB_FACTOR x o maior dente
    de dentro (um molar cortado pela borda é no máximo do tamanho de um molar)."""
    border = [touches_border(m) for m in masks]
    inner = [int(m.sum()) for m, b in zip(masks, border) if not b]
    if len(inner) < 3:
        return [False] * len(masks)
    lim = BORDER_BLOB_FACTOR * max(inner)
    return [b and int(m.sum()) > lim for m, b in zip(masks, border)]


def opposite_arch_flags(masks, upper):
    """
    Dentes da arcada oposta só aparecem no lado ABERTO do U, perto do centro (ali não existe dente da
    arcada certa: os molares ficam nos ramos). Também descarta pedaços pequenos cortados pela borda do
    lado aberto (os cantos da foto). Só age se houver dente central no lado dos incisivos, o que confirma
    a orientação esperada da foto.
    """
    n = len(masks)
    if n < 5:
        return [False] * n
    c = _centroids(masks)
    x, y = c[:, 0], c[:, 1]
    x0 = 0.5 * (np.percentile(x, 5) + np.percentile(x, 95))
    half_w = 0.5 * (np.percentile(x, 95) - np.percentile(x, 5))
    y0, y1 = y.min(), y.max()
    band = OPEN_SIDE_FRAC * (y1 - y0)
    central = np.abs(x - x0) < OPEN_CENTER_FRAC * half_w
    open_side = (y > y1 - band) if upper else (y < y0 + band)
    incisor_side = (y < y0 + band) if upper else (y > y1 - band)
    if not (central & incisor_side).any():
        return [False] * n
    h = masks[0].shape[0]
    on_open_border = np.array([bbox(m)[3] >= h - 4 if upper else bbox(m)[1] <= 3 for m in masks])
    areas = np.array([m.sum() for m in masks])
    small = on_open_border & (areas < OPEN_SMALL_FRAC * np.median(areas))
    return list((central & open_side) | small)


def off_arch_flags(masks):
    """
    Ajusta a parábola y = a x^2 + b x + c (robusta, pesos de Cauchy) nos centroides das máscaras fora da
    borda (peso igual: um bloco grande não entorta a curva) e descarta as que ficam a mais de
    CURVE_MAX_DIST tamanhos de dente da curva (dente/tecido solto no meio ou fora do U).
    """
    n = len(masks)
    if n < CURVE_MIN_MASKS:
        return [False] * n
    c = _centroids(masks)
    fit = np.array([not touches_border(m) for m in masks])
    if fit.sum() < CURVE_MIN_MASKS:
        fit[:] = True
    x, y = c[fit, 0], c[fit, 1]
    size = np.sqrt(np.median([m.sum() for m in masks]))
    coef = np.polyfit(x, y, 2)
    for _ in range(10):
        r = np.abs(np.polyval(coef, x) - y)
        coef = np.polyfit(x, y, 2, w=1.0 / np.maximum(1.0, r / (0.5 * size)))
    xs = np.linspace(c[:, 0].min() - size, c[:, 0].max() + size, 600)
    curve = np.stack([xs, np.polyval(coef, xs)], 1)
    dist = np.sqrt(((c[:, None, :] - curve[None]) ** 2).sum(-1)).min(1)
    far = dist > CURVE_MAX_DIST * size
    if (~far).sum() < CURVE_MIN_KEEP * n:
        return [False] * n
    return list(far)
