import numpy as np

from fase1_entrada.segmentacao.masks import bbox


def _vertical_relation(mask, x0, x1, cy, w_u, min_area_px):
    """'above'/'below' se a máscara é vizinha vertical relevante da região;
    None se for fragmento pequeno ou não houver sobreposição real em X."""
    if int((mask > 0).sum()) < min_area_px:
        return None
    a0, _, a1, _, _, cyo = bbox(mask)
    if min(x1, a1) - max(x0, a0) < 0.3 * w_u:
        return None
    return "above" if cyo < cy else "below"


def _shared_target(i, j, union, upper_masks, lower_masks):
    """Em que arcada o dente duplicado (upper[i] == lower[j]) fica: 'upper', 'lower' ou None (ambíguo)."""
    x0, _, x1, _, _, cy = bbox(union)
    w_u = max(1, x1 - x0)
    min_neighbor_area = 0.3 * int((union > 0).sum())
    has_above = any(_vertical_relation(m, x0, x1, cy, w_u, min_neighbor_area) == "above"
                    for k, m in enumerate(upper_masks) if k != i)
    has_below = any(_vertical_relation(m, x0, x1, cy, w_u, min_neighbor_area) == "below"
                    for k, m in enumerate(lower_masks) if k != j)
    if has_above and not has_below:
        return "lower"
    if has_below and not has_above:
        return "upper"
    return None


def resolve_shared_teeth(upper_masks, lower_masks):
    """
    Mesmo dente atribuído às duas arcadas (detecção duplicada): resolve pela
    vizinhança vertical usando as próprias máscaras —
      - se existe um dente ACIMA dele -> ele é INFERIOR
      - se existe um dente ABAIXO dele -> ele é SUPERIOR
    Regras de segurança:
      - vizinho só conta se tiver tamanho comparável ao do dente em conflito
        (fragmentos pequenos do próprio dente não contam);
      - se a decisão for ambígua, NADA é removido (o dente fica nas duas
        arcadas, como antes) — um dente nunca é perdido.
    Retorna (upper_masks, lower_masks, n_resolvidos).
    """
    consumed_u, consumed_l = set(), set()
    decisions = []  # (union, target)

    for i, mu in enumerate(upper_masks):
        for j, ml in enumerate(lower_masks):
            if i in consumed_u or j in consumed_l:
                continue
            inter = int(np.logical_and(mu > 0, ml > 0).sum())
            smaller = min(int((mu > 0).sum()), int((ml > 0).sum()))
            if inter == 0 or smaller == 0 or inter / smaller <= 0.5:
                continue
            union = np.maximum(mu, ml)
            target = _shared_target(i, j, union, upper_masks, lower_masks)
            if target is None:
                continue  # ambíguo: mantém as duas cópias, não perde o dente
            consumed_u.add(i)
            consumed_l.add(j)
            decisions.append((union, target))

    new_upper = [m for k, m in enumerate(upper_masks) if k not in consumed_u]
    new_lower = [m for k, m in enumerate(lower_masks) if k not in consumed_l]
    for union, target in decisions:
        (new_upper if target == "upper" else new_lower).append(union)
    return new_upper, new_lower, len(decisions)
