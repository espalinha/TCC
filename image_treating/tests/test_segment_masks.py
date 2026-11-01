import numpy as np
import pytest

import fase1_entrada.segmentacao.masks as st
from helpers import rect_mask


# ---------- fill_holes ----------

def test_fill_holes_fills_enclosed_hole():
    m = rect_mask((60, 60), 10, 40, 10, 40)
    m[20:30, 20:30] = 0
    assert int(st.fill_holes(m).sum()) == 30 * 30


def test_fill_holes_keeps_hole_open_to_the_border():
    m = rect_mask((60, 60), 0, 50, 0, 50)
    m[20:30, 0:30] = 0                       # entalhe que chega na borda: não é buraco
    assert st.fill_holes(m)[25, 25] == 0


def test_fill_holes_hole_touching_image_corner_region_is_filled():
    m = np.ones((40, 40), np.uint8)          # máscara ocupando a imagem toda, com buraco no meio
    m[15:25, 15:25] = 0
    assert st.fill_holes(m).sum() == 40 * 40


def test_fill_holes_empty_mask_returns_zeros():
    out = st.fill_holes(np.zeros((20, 30), np.uint8))
    assert out.shape == (20, 30) and out.sum() == 0


def test_fill_holes_binarizes_any_positive_value():
    m = np.zeros((20, 20), np.uint8)
    m[5:15, 5:15] = 7
    out = st.fill_holes(m)
    assert set(np.unique(out)) == {0, 1}


# ---------- clean_mask ----------

def test_clean_mask_rejects_below_threshold_values():
    assert st.clean_mask(np.full((50, 50), 0.4, np.float32), min_area=1) is None
    assert st.clean_mask(np.full((50, 50), 0.5, np.float32), min_area=1) is None   # > 0.5 estrito


def test_clean_mask_rejects_too_small_area():
    assert st.clean_mask(rect_mask((100, 100), 10, 15, 10, 15).astype(np.float32), min_area=100) is None


def test_clean_mask_rejects_empty_even_with_zero_min_area():
    assert st.clean_mask(np.zeros((50, 50), np.float32), min_area=0) is None


def test_clean_mask_keeps_only_largest_component():
    m = rect_mask((200, 200), 20, 100, 20, 100)
    m[150:170, 150:170] = 1
    out = st.clean_mask(m.astype(np.float32), min_area=100)
    assert out[50, 50] == 1 and out[160, 160] == 0


def test_clean_mask_rejects_when_largest_component_is_below_min_area():
    m = np.zeros((200, 200), np.float32)
    for k in range(5):
        m[10 + 30 * k:20 + 30 * k, 10:20] = 1     # 5 blobs de 100 px; soma 500
    assert st.clean_mask(m, min_area=300) is None


def test_clean_mask_fills_holes_and_returns_binary_uint8():
    m = rect_mask((200, 200), 40, 160, 40, 160).astype(np.float32)
    m[90:110, 90:110] = 0
    out = st.clean_mask(m, min_area=100)
    assert out[100, 100] == 1 and out.dtype == np.uint8 and set(np.unique(out)) <= {0, 1}


def test_clean_mask_accepts_probability_masks():
    m = np.zeros((100, 100), np.float32)
    m[20:80, 20:80] = 0.9
    m[0:10, 0:10] = 0.3
    out = st.clean_mask(m, min_area=100)
    assert out[50, 50] == 1 and out[5, 5] == 0


def test_clean_mask_closes_thin_slit_at_tooth_scale():
    m = rect_mask((300, 300), 50, 250, 50, 250)
    m[148:152, 50:200] = 0                   # fissura de 4 px num dente de 200 px (kernel ~12 px)
    assert st.clean_mask(m.astype(np.float32), min_area=100)[150, 100] == 1


# ---------- complete_convex ----------

def test_complete_convex_fills_small_notch():
    m = rect_mask((100, 100), 10, 90, 10, 90)
    m[10:25, 40:55] = 0                      # canto comido (solidez ~0.96)
    assert st.complete_convex(m)[15, 47] == 1


def test_complete_convex_leaves_very_concave_shape():
    m = rect_mask((100, 100), 10, 90, 10, 25) | rect_mask((100, 100), 75, 90, 10, 90)  # "L"
    assert st.solidity(m)[0] < st.HULL_MIN_SOLIDITY
    assert np.array_equal(st.complete_convex(m), m)


# ---------- absorb_defects ----------

def test_absorb_defects_merges_defect_on_tooth_and_fills_hole():
    tooth = rect_mask((100, 100), 10, 90, 10, 90)
    tooth[30:70, 30:70] = 0                  # restauração escura que o YOLO marcou como cárie, não dente
    defect = rect_mask((100, 100), 35, 75, 35, 75)   # encosta no dente
    out = st.absorb_defects([tooth], [defect])
    assert out[0][50, 50] == 1 and out[0][32, 32] == 1


def test_absorb_defects_ignores_defect_far_from_teeth():
    tooth = rect_mask((100, 200), 10, 50, 10, 50)
    far = rect_mask((100, 200), 10, 50, 150, 190)
    out = st.absorb_defects([tooth], [far])
    assert np.array_equal(out[0], tooth)


# ---------- arch_mask ----------

def test_arch_mask_empty_list_returns_zeros_with_image_shape():
    out = st.arch_mask([], (30, 40, 3))
    assert out.shape == (30, 40) and out.sum() == 0


def test_arch_mask_closes_gap_between_neighbouring_teeth():
    a = rect_mask((200, 300), 60, 140, 50, 120)
    b = rect_mask((200, 300), 60, 140, 128, 200)   # vão de 8 px; dente ~75 px -> kernel ~13 px
    assert st.arch_mask([a, b], (200, 300))[100, 124] == 1


def test_arch_mask_keeps_far_apart_teeth_separate():
    a = rect_mask((200, 400), 60, 140, 20, 90)
    b = rect_mask((200, 400), 60, 140, 300, 370)
    assert st.arch_mask([a, b], (200, 400))[100, 200] == 0


# ---------- remove_outlier_masks ----------

def _row(xs, shape=(200, 2000)):
    return [rect_mask(shape, 90, 110, x - 5, x + 5) for x in xs]


def test_remove_outliers_needs_at_least_five_masks():
    masks = _row([100, 150, 200, 1500])
    kept, n = st.remove_outlier_masks(masks)
    assert kept is masks and n == 0


def test_remove_outliers_drops_isolated_blob():
    kept, n = st.remove_outlier_masks(_row([100, 150, 200, 250, 300, 1500]))
    assert n == 1 and len(kept) == 5


def test_remove_outliers_keeps_homogeneous_group():
    kept, n = st.remove_outlier_masks(_row([100, 150, 200, 250, 300]))
    assert n == 0 and len(kept) == 5


def test_remove_outliers_exactly_five_masks_are_evaluated():
    kept, n = st.remove_outlier_masks(_row([100, 150, 200, 250, 1900]))
    assert n == 1


def test_remove_outliers_identical_centroids_are_all_kept():
    masks = [rect_mask((100, 100), 40, 60, 40, 60) for _ in range(5)]
    kept, n = st.remove_outlier_masks(masks)
    assert n == 0 and len(kept) == 5


def test_remove_outliers_preserves_order():
    masks = _row([100, 150, 200, 250, 300, 1500])
    kept, _ = st.remove_outlier_masks(masks)
    assert all(a is b for a, b in zip(kept, masks[:5]))


# ---------- combine_masks / bbox ----------

def test_combine_masks_empty_list_returns_zeros_with_image_shape():
    out = st.combine_masks([], (30, 40, 3))
    assert out.shape == (30, 40) and out.dtype == np.uint8 and out.sum() == 0


def test_combine_masks_is_union_and_ignores_channels():
    a = rect_mask((50, 50), 0, 20, 0, 20)
    b = rect_mask((50, 50), 10, 30, 10, 30)
    out = st.combine_masks([a, b], (50, 50, 3))
    assert int(out.sum()) == 400 + 400 - 100


def test_bbox_returns_extents_and_centroid():
    m = rect_mask((100, 100), 10, 21, 30, 41)
    assert st.bbox(m) == (30, 10, 40, 20, 35.0, 15.0)


def test_bbox_of_empty_mask_raises():
    with pytest.raises(ValueError):
        st.bbox(np.zeros((10, 10), np.uint8))
