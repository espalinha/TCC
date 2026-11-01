import cv2
import numpy as np
import pytest

from helpers import FakeModel, rect_mask
from fase1_entrada.segmentacao import color, occlusal, workflow
from fase1_entrada.segmentacao.detector import YoloDetector
from fase1_entrada.segmentacao.state import ViewState
from fase1_entrada.segmentacao.views import VIEW_FRONT, VIEW_LOWER_OCC, VIEW_UPPER_OCC

SHAPE = (600, 900)


def _disk(cx, cy, r, shape=SHAPE):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(cx), int(cy)), int(r), 1, -1)
    return m


def _u_arch(upper, n=12, shape=SHAPE):
    """Dentes (discos) num U: superior com o vértice (incisivos) em cima, inferior com o vértice embaixo."""
    t = np.linspace(-1, 1, n)
    xs = 450 + 300 * t
    ys = 120 + 330 * t ** 2 if upper else 480 - 330 * t ** 2
    return [_disk(x, y, 28, shape) for x, y in zip(xs, ys)]


# ---------- arcada oposta ----------

@pytest.mark.parametrize("upper", [True, False])
def test_opposite_arch_row_in_open_side_is_flagged(upper):
    arch = _u_arch(upper)
    y_open = 560 if upper else 40        # fileira da arcada oposta no lado aberto, no centro
    opposite = [_disk(x, y_open, 25) for x in (390, 450, 510)]
    flags = occlusal.opposite_arch_flags(arch + opposite, upper)
    assert flags[-3:] == [True] * 3 and not any(flags[:-3])


def test_opposite_arch_keeps_real_incisors_cut_by_the_border():
    arch = _u_arch(True)
    arch[5] = rect_mask(SHAPE, 0, 60, 410, 445)     # incisivos reais encostados na borda de cima
    arch[6] = rect_mask(SHAPE, 0, 60, 455, 490)
    assert not any(occlusal.opposite_arch_flags(arch, upper=True))


def test_opposite_arch_does_nothing_when_orientation_is_unexpected():
    arch = _u_arch(False)                            # U de inferior numa oclusal "superior"
    assert not any(occlusal.opposite_arch_flags(arch, upper=True))


def test_opposite_arch_drops_small_piece_on_open_side_border():
    arch = _u_arch(False)
    corner = rect_mask(SHAPE, 0, 12, 20, 45)         # pedacinho no canto de cima (lado aberto do inferior)
    flags = occlusal.opposite_arch_flags(arch + [corner], upper=False)
    assert flags[-1] and not any(flags[:-1])


# ---------- curva da arcada ----------

def test_off_arch_flags_tooth_inside_the_u():
    arch = _u_arch(True)
    stray = _disk(450, 380, 28)                      # solto no meio do palato
    flags = occlusal.off_arch_flags(arch + [stray])
    assert flags[-1] and not any(flags[:-1])


def test_off_arch_not_bent_by_big_border_blob():
    arch = _u_arch(False)
    blob = rect_mask(SHAPE, 250, 600, 820, 900)      # bloco grande na borda: fica fora do ajuste
    assert not any(occlusal.off_arch_flags(arch + [blob])[:-1])


def test_off_arch_needs_enough_masks():
    assert occlusal.off_arch_flags(_u_arch(True, n=5)) == [False] * 5


# ---------- borda e forma ----------

def test_border_blob_bigger_than_largest_inner_tooth_is_flagged():
    arch = _u_arch(False)
    blob = rect_mask(SHAPE, 300, 600, 800, 900)
    flags = occlusal.border_blob_flags(arch + [blob])
    assert flags[-1] and not any(flags[:-1])


def test_tissue_shape_flags_curly_mask_but_keeps_confident_notched_tooth():
    crescent = _disk(200, 200, 60) & (1 - _disk(235, 200, 55))       # tecido "enrolado": solidez baixa
    notched = rect_mask(SHAPE, 100, 200, 400, 500)
    notched[100:180, 420:480] = 0                                    # dente com entalhe grande (solidez ~0.5)
    assert occlusal.tissue_shape_flags([crescent, notched], [0.5, 0.95]) == [True, False]
    assert occlusal.tissue_shape_flags([notched], [0.5]) == [True]


# ---------- cor ----------

def _photo_with(a_stars, masks):
    """Foto em Lab: L e b* fixos, a* (vermelho) dado por máscara."""
    lab = np.full((*SHAPE, 3), (200, 128, 140), np.uint8)
    for m, a in zip(masks, a_stars):
        lab[m > 0, 1] = a
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def test_red_tissue_low_confidence_is_flagged_but_confident_gold_crown_is_kept():
    teeth = [_disk(100 + 120 * k, 300, 40) for k in range(7)]
    img = _photo_with([134] * 4 + [146, 146, 175], teeth)   # +12: lábio (conf baixa) / ouro (conf alta); +41
    flags = color.red_tissue_flags(teeth, [0.9, 0.9, 0.9, 0.9, 0.3, 0.95, 0.95], img)
    assert flags == [False, False, False, False, True, False, True]


def test_red_tissue_needs_enough_masks():
    teeth = [_disk(100, 100, 20), _disk(200, 100, 20)]
    assert color.red_tissue_flags(teeth, [0.2, 0.2], np.zeros((*SHAPE, 3), np.uint8)) == [False, False]


# ---------- workflow ----------

@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.jpg"
    cv2.imwrite(str(path), np.full((*SHAPE, 3), 180, np.uint8))
    return str(path)


def _run(model, view, path, stages, conf=0.18):
    return workflow.run(ViewState(view, path, conf=conf), stages(YoloDetector(model, 640, conf)))


def test_occlusal_workflow_drops_opposite_arch(photo):
    arch = _u_arch(False)
    opposite = [_disk(x, 40, 25) for x in (390, 450, 510)]
    state = _run(FakeModel(arch + opposite), VIEW_LOWER_OCC, photo, workflow.occlusal)
    assert len(state.lower) == len(arch)


def test_defect_mask_fills_black_hole_in_tooth(photo):
    tooth = rect_mask(SHAPE, 200, 400, 100, 300)
    tooth[260:340, 160:240] = 0
    defect = rect_mask(SHAPE, 270, 330, 170, 230)
    teeth = [tooth] + [rect_mask(SHAPE, 200, 400, x, x + 150) for x in (350, 550, 750)]
    state = _run(FakeModel(teeth + [defect], [0.9] * 5, [3, 3, 3, 3, 0]), VIEW_FRONT, photo, workflow.lateral)
    merged = (state.upper + state.lower)
    assert any(m[300, 200] == 1 for m in merged)


def test_lateral_ignores_teeth_below_base_confidence(photo):
    teeth = [rect_mask(SHAPE, 200, 400, x, x + 150) for x in (100, 300, 500)]
    state = _run(FakeModel(teeth, [0.9, 0.9, 0.1]), VIEW_FRONT, photo, workflow.lateral)
    assert len(state.upper) + len(state.lower) == 2


def test_occlusal_upper_view_puts_masks_in_upper(photo):
    state = _run(FakeModel(_u_arch(True)), VIEW_UPPER_OCC, photo, workflow.occlusal)
    assert len(state.upper) == 12 and state.lower == []
