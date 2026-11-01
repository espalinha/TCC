import fase1_entrada.segmentacao.arch_split as split
import fase1_entrada.segmentacao.shared_teeth as shared
import fase1_entrada.segmentacao.straddling as straddling
import numpy as np
import pytest
from helpers import centroid_y, rect_mask, teeth_rows

SHAPE = (400, 800)


# ---------- split_upper_lower ----------


def test_split_without_masks():
    assert split.split_upper_lower([], 400) == ([], [], None)


def test_split_ignores_empty_masks():
    assert split.split_upper_lower([np.zeros(SHAPE, np.uint8)], 400) == ([], [], None)


@pytest.mark.parametrize(
    "y0, goes_upper", [(20, True), (150, True), (250, False), (330, False)]
)
def test_split_single_mask_goes_by_image_half(y0, goes_upper):
    m = rect_mask(SHAPE, y0, y0 + 40, 100, 140)
    upper, lower, boundary = split.split_upper_lower([m], 400)
    assert boundary is None
    assert (len(upper), len(lower)) == ((1, 0) if goes_upper else (0, 1))


def test_split_two_rows_into_two_arches():
    up, lo = teeth_rows()
    upper, lower, boundary = split.split_upper_lower(up + lo, 400)
    assert len(upper) == 4 and len(lower) == 4
    assert all(centroid_y(m) < 200 for m in upper) and all(
        centroid_y(m) > 200 for m in lower
    )
    mean, v, thr = boundary
    assert 160 < mean[1] + thr * v[1] < 240  # fronteira (em y) entre as duas fileiras


def test_split_is_independent_of_input_order():
    up, lo = teeth_rows()
    mixed = [lo[0], up[0], lo[1], up[1], up[2], lo[2], lo[3], up[3]]
    upper, lower, _ = split.split_upper_lower(mixed, 400)
    assert len(upper) == 4 and len(lower) == 4


def test_split_single_arch_with_many_teeth_is_not_divided():
    up, _ = teeth_rows()
    upper, lower, boundary = split.split_upper_lower(up, 400)
    assert (len(upper), len(lower), boundary) == (4, 0, None)


def test_split_single_arch_in_lower_half_goes_to_lower():
    _, lo = teeth_rows()
    upper, lower, boundary = split.split_upper_lower(lo, 400)
    assert (len(upper), len(lower), boundary) == (0, 4, None)


def test_split_with_vertical_column_uses_y_axis():
    up = [rect_mask(SHAPE, 80 + 10 * k, 140 + 10 * k, 280, 320) for k in range(2)]
    lo = [rect_mask(SHAPE, 280 + 10 * k, 340 + 10 * k, 280, 320) for k in range(2)]
    upper, lower, boundary = split.split_upper_lower(up + lo, 400)
    assert (len(upper), len(lower)) == (2, 2)
    assert list(boundary[1]) == [0.0, 1.0]


def test_split_boundary_vector_points_downwards():
    up, lo = teeth_rows()
    _, _, (mean, v, thr) = split.split_upper_lower(up + lo, 400)
    assert v[1] > 0 and abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_split_tilted_occlusal_plane_still_separates():
    up, lo = [], []
    for i, x in enumerate((100, 250, 400, 550)):
        up.append(rect_mask(SHAPE, 90 + 12 * i, 150 + 12 * i, x, x + 40))
        lo.append(rect_mask(SHAPE, 250 + 12 * i, 310 + 12 * i, x, x + 40))
    upper, lower, _ = split.split_upper_lower(up + lo, 400)
    assert len(upper) == 4 and len(lower) == 4


def test_split_unbalanced_clusters_cut_at_median():
    row = [
        rect_mask(SHAPE, 290, 350, x, x + 40)
        for x in (40, 130, 220, 310, 400, 490, 580)
    ]
    lone = rect_mask(SHAPE, 40, 100, 300, 340)
    upper, lower, boundary = split.split_upper_lower(row + [lone], 400)
    assert boundary is not None
    assert len(upper) + len(lower) == 8  # nenhum dente se perde


def test_split_never_loses_or_duplicates_teeth():
    up, lo = teeth_rows(xs=(60, 160, 260, 360, 460, 560, 660))
    masks = up + lo
    upper, lower, _ = split.split_upper_lower(masks, 400)
    assert len(upper) + len(lower) == len(masks)
    ids = {id(m) for m in upper} | {id(m) for m in lower}
    assert ids == {id(m) for m in masks}


# ---------- _vertical_relation ----------


def test_vertical_relation_above_below_and_none():
    nb = rect_mask(SHAPE, 100, 160, 100, 140)
    assert shared._vertical_relation(nb, 100, 139, 230, 39, 100) == "above"
    assert shared._vertical_relation(nb, 100, 139, 50, 39, 100) == "below"


def test_vertical_relation_ignores_small_fragment():
    nb = rect_mask(SHAPE, 100, 110, 100, 110)
    assert shared._vertical_relation(nb, 100, 139, 230, 39, 1000) is None


def test_vertical_relation_requires_x_overlap():
    nb = rect_mask(SHAPE, 100, 160, 400, 440)
    assert shared._vertical_relation(nb, 100, 139, 230, 39, 100) is None


# ---------- resolve_shared_teeth ----------


def _dup():
    return rect_mask(SHAPE, 200, 260, 100, 140), rect_mask(SHAPE, 200, 260, 100, 140)


def test_resolve_shared_without_overlap_changes_nothing():
    up = [rect_mask(SHAPE, 50, 110, 100, 140)]
    lo = [rect_mask(SHAPE, 300, 360, 100, 140)]
    new_u, new_l, n = shared.resolve_shared_teeth(up, lo)
    assert (len(new_u), len(new_l), n) == (1, 1, 0)


def test_resolve_shared_ambiguous_without_neighbors_keeps_both_copies():
    a, b = _dup()
    new_u, new_l, n = shared.resolve_shared_teeth([a], [b])
    assert (len(new_u), len(new_l), n) == (1, 1, 0)


def test_resolve_shared_tooth_with_upper_neighbor_above_becomes_lower():
    a, b = _dup()
    above = rect_mask(SHAPE, 100, 160, 100, 140)
    new_u, new_l, n = shared.resolve_shared_teeth([a, above], [b])
    assert n == 1 and len(new_u) == 1 and len(new_l) == 1
    assert (
        centroid_y(new_u[0]) < 200 and centroid_y(new_l[0]) > 200
    )  # sobra so o vizinho em cima; o dente vai pra baixo


def test_resolve_shared_tooth_with_lower_neighbor_below_becomes_upper():
    a, b = _dup()
    below = rect_mask(SHAPE, 300, 360, 100, 140)
    new_u, new_l, n = shared.resolve_shared_teeth([a], [b, below])
    assert n == 1 and len(new_u) == 1 and len(new_l) == 1
    assert centroid_y(new_u[0]) > 200 - 1 and centroid_y(new_l[0]) > 300


def test_resolve_shared_with_neighbors_on_both_sides_is_ambiguous():
    a, b = _dup()
    above = rect_mask(SHAPE, 100, 160, 100, 140)
    below = rect_mask(SHAPE, 300, 360, 100, 140)
    new_u, new_l, n = shared.resolve_shared_teeth([a, above], [b, below])
    assert (len(new_u), len(new_l), n) == (2, 2, 0)


def test_resolve_shared_ignores_small_neighbor_fragments():
    a, b = _dup()
    fragment = rect_mask(SHAPE, 100, 105, 100, 105)
    new_u, new_l, n = shared.resolve_shared_teeth([a, fragment], [b])
    assert n == 0 and len(new_u) == 2 and len(new_l) == 1


def test_resolve_shared_ignores_neighbor_without_x_overlap():
    a, b = _dup()
    far = rect_mask(SHAPE, 100, 160, 500, 540)
    assert shared.resolve_shared_teeth([a, far], [b])[2] == 0


def test_resolve_shared_requires_overlap_above_half_of_smaller_mask():
    a = rect_mask(SHAPE, 200, 260, 100, 140)
    b = rect_mask(SHAPE, 200, 260, 120, 160)  # 50% exato: nao conta como duplicata
    above = rect_mask(SHAPE, 100, 160, 100, 140)
    assert shared.resolve_shared_teeth([a, above], [b])[2] == 0


def test_resolve_shared_result_is_union_of_the_two_copies():
    a = rect_mask(SHAPE, 200, 260, 100, 140)
    b = rect_mask(SHAPE, 205, 265, 100, 140)
    above = rect_mask(SHAPE, 100, 160, 100, 140)
    _, new_l, n = shared.resolve_shared_teeth([a, above], [b])
    assert n == 1
    assert int(new_l[0].sum()) == int(np.maximum(a, b).sum())


# ---------- _valley_cut ----------


def test_valley_cut_finds_valley_between_two_lobes():
    rng = np.random.RandomState(0)
    proj = np.concatenate([rng.normal(-100, 10, 500), rng.normal(100, 10, 500)])
    cut = straddling._valley_cut(proj, 999.0)
    assert -80 < cut < 80  # cai no vale (borda do trecho vazio), entre os dois lobos
    assert cut != 999.0


def test_valley_cut_returns_default_for_unimodal_profile():
    proj = np.random.RandomState(1).normal(0, 20, 1000)
    assert straddling._valley_cut(proj, 7.5) == 7.5


def test_valley_cut_returns_default_for_constant_profile():
    assert straddling._valley_cut(np.full(100, 3.0), 1.25) == 1.25


def test_valley_cut_returns_default_when_lobes_are_not_separated_enough():
    rng = np.random.RandomState(2)
    proj = np.concatenate([rng.normal(0, 10, 500), rng.normal(15, 10, 500)])
    assert straddling._valley_cut(proj, 42.0) == 42.0


# ---------- separate_straddling ----------

# (media, eixo, limiar): a projecao e relativa a media, entao thr=0 equivale a y=200 aqui
BOUNDARY = (np.array([120.0, 200.0]), np.array([0.0, 1.0]), 0.0)


def test_separate_straddling_without_boundary_is_noop():
    up, lo = teeth_rows()
    new_u, new_l, n = straddling.separate_straddling(up, lo, None, 50)
    assert new_u is up and new_l is lo and n == 0


def test_separate_straddling_keeps_well_placed_teeth():
    up, lo = teeth_rows()
    new_u, new_l, n = straddling.separate_straddling(up, lo, BOUNDARY, 50)
    assert (len(new_u), len(new_l), n) == (4, 4, 0)


def test_separate_straddling_reassigns_tooth_in_wrong_list_by_centroid():
    wrong = rect_mask(
        SHAPE, 280, 340, 100, 140
    )  # abaixo da fronteira mas na lista superior
    new_u, new_l, n = straddling.separate_straddling([wrong], [], BOUNDARY, 50)
    assert (len(new_u), len(new_l), n) == (0, 1, 0)


def test_separate_straddling_cuts_fused_upper_lower_blob():
    fused = rect_mask(SHAPE, 100, 300, 100, 140)  # atravessa a fronteira (y=200)
    new_u, new_l, n = straddling.separate_straddling([fused], [], BOUNDARY, 50)
    assert n == 1 and len(new_u) == 1 and len(new_l) == 1
    assert centroid_y(new_u[0]) < 200 < centroid_y(new_l[0])
    assert int(new_u[0].sum()) + int(new_l[0].sum()) >= int(fused.sum()) - 40


def test_separate_straddling_mostly_one_side_is_not_cut():
    mostly_up = rect_mask(SHAPE, 100, 220, 100, 140)  # 17% abaixo (< 30%)
    new_u, new_l, n = straddling.separate_straddling([mostly_up], [], BOUNDARY, 50)
    assert (len(new_u), len(new_l), n) == (1, 0, 0)


def test_separate_straddling_does_not_cut_when_a_piece_is_below_min_area():
    fused = rect_mask(SHAPE, 100, 300, 100, 140)
    new_u, new_l, n = straddling.separate_straddling([fused], [], BOUNDARY, 10**6)
    assert n == 0 and len(new_u) + len(new_l) == 1


def test_separate_straddling_drops_empty_masks():
    new_u, new_l, n = straddling.separate_straddling(
        [np.zeros(SHAPE, np.uint8)], [], BOUNDARY, 50
    )
    assert (new_u, new_l, n) == ([], [], 0)


def test_separate_straddling_reports_cuts(capsys):
    straddling.separate_straddling(
        [rect_mask(SHAPE, 100, 300, 100, 140)], [], BOUNDARY, 50
    )
    assert "separada" in capsys.readouterr().out
