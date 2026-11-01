import fase1_entrada.segmentacao.render as st
import numpy as np
import pytest
from fase1_entrada.segmentacao.confidence import adaptive_conf_threshold
from helpers import rect_mask


def _photo(h, w, value=200):
    return np.full((h, w, 3), value, np.uint8)


# ---------- apply_mask_square ----------


@pytest.mark.parametrize("h, w", [(100, 200), (200, 100), (150, 150), (101, 203)])
def test_apply_mask_square_always_outputs_square_rgb(h, w):
    out = st.apply_mask_square(_photo(h, w), np.ones((h, w), np.uint8))
    assert out.shape == (st.OUTPUT_SIZE, st.OUTPUT_SIZE, 3) and out.dtype == np.uint8


def test_apply_mask_square_custom_size():
    assert st.apply_mask_square(
        _photo(50, 80), np.ones((50, 80), np.uint8), out_size=64
    ).shape == (64, 64, 3)


def test_apply_mask_square_background_is_black_outside_mask():
    mask = rect_mask((100, 100), 40, 60, 40, 60)
    out = st.apply_mask_square(_photo(100, 100), mask, out_size=100)
    assert out[50, 50].tolist() == [200, 200, 200] and out[5, 5].tolist() == [0, 0, 0]


def test_apply_mask_square_empty_mask_gives_black_image():
    out = st.apply_mask_square(
        _photo(60, 90), np.zeros((60, 90), np.uint8), out_size=32
    )
    assert out.sum() == 0


def test_apply_mask_square_pads_wide_image_vertically_centered():
    out = st.apply_mask_square(
        _photo(100, 200), np.ones((100, 200), np.uint8), out_size=200
    )
    rows = np.where(out.sum(axis=(1, 2)) > 0)[0]
    assert (
        rows.min() == 50 and rows.max() == 149
    )  # 50 px de borda preta em cima e embaixo


def test_apply_mask_square_pads_tall_image_horizontally_centered():
    out = st.apply_mask_square(
        _photo(200, 100), np.ones((200, 100), np.uint8), out_size=200
    )
    cols = np.where(out.sum(axis=(0, 2)) > 0)[0]
    assert cols.min() == 50 and cols.max() == 149


def test_apply_mask_square_does_not_mutate_inputs():
    img, mask = _photo(40, 60), np.ones((40, 60), np.uint8)
    img0, mask0 = img.copy(), mask.copy()
    st.apply_mask_square(img, mask)
    assert np.array_equal(img, img0) and np.array_equal(mask, mask0)


# ---------- mask_to_rgba ----------


def test_mask_to_rgba_alpha_follows_mask():
    mask = rect_mask((20, 30), 5, 10, 5, 10)
    rgba = st.mask_to_rgba(_photo(20, 30), mask)
    assert rgba.shape == (20, 30, 4)
    assert rgba[7, 7, 3] == 255 and rgba[0, 0, 3] == 0
    assert rgba[7, 7, :3].tolist() == [200, 200, 200]


# ---------- draw_overlay ----------


def test_draw_overlay_keeps_size_when_small_and_does_not_mutate_input():
    img = _photo(300, 400)
    img0 = img.copy()
    out = st.draw_overlay(img, [rect_mask((300, 400), 100, 150, 100, 150)], [])
    assert out.shape == img.shape and np.array_equal(img, img0)


def test_draw_overlay_downscales_to_max_dim():
    out = st.draw_overlay(_photo(1000, 3200), [], [], max_dim=1600)
    assert out.shape[:2] == (500, 1600)


def test_draw_overlay_colors_upper_and_lower_differently():
    up = rect_mask((300, 400), 50, 100, 50, 100)
    lo = rect_mask((300, 400), 200, 250, 50, 100)
    out = st.draw_overlay(_photo(300, 400, 0), [up], [lo])
    assert out[60, 90].tolist() != out[210, 90].tolist()
    assert out[60, 90, 1] > out[60, 90, 0]  # verde (BGR) domina no superior


def test_draw_overlay_without_masks_only_adds_legend():
    img = _photo(300, 400, 0)
    out = st.draw_overlay(img, [], [])
    assert out[200:, :].sum() == 0 and out[:60, :].sum() > 0


# ---------- adaptive_conf_threshold ----------


def test_adaptive_threshold_needs_at_least_four_detections():
    assert adaptive_conf_threshold(np.array([0.9, 0.1, 0.8]), 0.18) == 0.18


def test_adaptive_threshold_homogeneous_confidences_use_base():
    assert (
        adaptive_conf_threshold(np.array([0.80, 0.82, 0.85, 0.88, 0.90]), 0.18) == 0.18
    )


def test_adaptive_threshold_separates_minority_of_false_positives():
    confs = np.array([0.90, 0.92, 0.88, 0.91, 0.89, 0.30, 0.25])
    thr = adaptive_conf_threshold(confs, 0.18)
    assert (
        0.4 < thr <= 0.85 * np.median(confs)
    )  # corta os falsos positivos, nunca o grupo principal


def test_adaptive_threshold_low_majority_is_not_cut():
    confs = np.array([0.30, 0.25, 0.28, 0.27, 0.90, 0.92])
    assert adaptive_conf_threshold(confs, 0.18) == 0.18


def test_adaptive_threshold_never_below_base_conf():
    confs = np.array([0.90, 0.92, 0.88, 0.91, 0.89, 0.30, 0.25])
    assert adaptive_conf_threshold(confs, 0.7) >= 0.7
