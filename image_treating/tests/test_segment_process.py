import os

import cv2
import numpy as np
import pytest

from helpers import FakeModel, rect_mask, teeth_rows
from fase1_entrada.segmentacao import cli, workflow
from fase1_entrada.segmentacao.detector import DETECT_MIN_CONF, YoloDetector
from fase1_entrada.segmentacao.render import OUTPUT_SIZE
from fase1_entrada.segmentacao.state import ViewState
from fase1_entrada.segmentacao.views import VIEW_FRONT, VIEW_LEFT, VIEW_LOWER_OCC, VIEW_UPPER_OCC

SHAPE = (400, 800)


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "foto.jpg"
    cv2.imwrite(str(path), np.full((*SHAPE, 3), 180, np.uint8))
    return str(path)


def _run(model, view, path, stages, imgsz=640, conf=0.2, device=None):
    detector = YoloDetector(model, imgsz, conf, device)
    return workflow.run(ViewState(view, path, conf=conf), stages(detector))


def lateral(model, photo, view=VIEW_FRONT, **kw):
    return _run(model, view, photo, workflow.lateral, **kw)


def occlusal(model, photo, view=VIEW_UPPER_OCC, **kw):
    return _run(model, view, photo, workflow.occlusal, **kw)


# ---------- workflow lateral ----------

def test_lateral_unreadable_image_stops(tmp_path):
    state = lateral(FakeModel([]), str(tmp_path / "nao_existe.jpg"))
    assert state.stopped and state.img is None


def test_lateral_without_detections(photo):
    state = lateral(FakeModel([]), photo)
    assert state.img is not None and not state.stopped
    assert (state.upper, state.lower, len(state.confs), state.n_outliers) == ([], [], 0, 0)


def test_lateral_splits_arches(photo):
    up, lo = teeth_rows(SHAPE)
    state = lateral(FakeModel(up + lo), photo)
    assert (len(state.upper), len(state.lower), len(state.confs)) == (4, 4, 8)


def test_detector_passes_expected_predict_arguments(photo):
    model = FakeModel([])
    lateral(model, photo, view=VIEW_LEFT, imgsz=1920, conf=0.18, device="cpu")
    kw = model.calls[0]
    assert kw["source"] == photo and kw["imgsz"] == 1920 and kw["conf"] == DETECT_MIN_CONF
    assert "classes" not in kw and kw["retina_masks"] is False and kw["device"] == "cpu"


def test_detector_omits_device_when_none(photo):
    model = FakeModel([])
    lateral(model, photo, view=VIEW_LEFT)
    assert "device" not in model.calls[0]


def test_lateral_resizes_masks_to_image_size(photo):
    small = rect_mask((200, 400), 40, 100, 50, 150)       # metade da resolucao
    state = lateral(FakeModel([small]), photo)
    assert (state.upper + state.lower)[0].shape == SHAPE


def test_lateral_discards_masks_below_min_area(photo):
    tiny = rect_mask(SHAPE, 10, 14, 10, 14)                # 16 px < 0.0002 * 320000 = 64 px
    state = lateral(FakeModel([tiny]), photo)
    assert state.upper == [] and state.lower == []


def test_lateral_drops_isolated_outlier(photo, capsys):
    row = [rect_mask(SHAPE, 300, 360, x, x + 40) for x in (60, 120, 180, 240, 300)]
    far = rect_mask(SHAPE, 20, 80, 740, 780)
    state = lateral(FakeModel(row + [far]), photo)
    assert state.n_outliers == 1 and len(state.upper) + len(state.lower) == 5
    assert "isolada" in capsys.readouterr().out


# ---------- workflow oclusal ----------

def _occlusal_teeth():
    return [rect_mask(SHAPE, 150, 250, x, x + 60) for x in (100, 180, 260, 340, 420)]


@pytest.mark.parametrize("view, side", [(VIEW_UPPER_OCC, "upper"), (VIEW_LOWER_OCC, "lower")])
def test_occlusal_is_never_split(photo, view, side):
    state = occlusal(FakeModel(_occlusal_teeth()), photo, view=view)
    assert (len(state.upper), len(state.lower)) == ((5, 0) if side == "upper" else (0, 5))


def test_occlusal_unreadable_image(tmp_path):
    assert occlusal(FakeModel([]), str(tmp_path / "x.jpg")).stopped


def test_occlusal_without_detections_stops(photo):
    state = occlusal(FakeModel([]), photo)
    assert state.img is not None and state.stopped


def test_occlusal_filters_low_confidence_false_positives(photo):
    teeth = _occlusal_teeth() + [rect_mask(SHAPE, 150, 250, 500, 560), rect_mask(SHAPE, 150, 250, 580, 640)]
    confs = [0.9, 0.92, 0.88, 0.91, 0.89, 0.2, 0.25]
    filtered = occlusal(FakeModel(teeth, confs), photo, conf=0.18)
    unfiltered = occlusal(FakeModel(teeth, [0.9] * 7), photo, conf=0.18)
    assert len(filtered.upper) < len(unfiltered.upper)


def test_occlusal_all_masks_too_small_stops(photo):
    tiny = [rect_mask(SHAPE, 10, 13, 10 + 20 * k, 13 + 20 * k) for k in range(3)]
    assert occlusal(FakeModel(tiny), photo).stopped


def test_run_stops_calling_stages_after_stopped(photo):
    calls = []
    stages = [lambda s: s.__class__(**{**s.__dict__, "stopped": True}), lambda s: calls.append(1) or s]
    workflow.run(ViewState(VIEW_FRONT, photo), stages)
    assert calls == []


# ---------- cli (ponta a ponta com YOLO falso) ----------

def _write_photo(path, shape=SHAPE):
    cv2.imwrite(str(path), np.full((*shape, 3), 180, np.uint8))


def _run_main(monkeypatch, argv, model):
    monkeypatch.setattr(cli, "load_yolo", lambda weights: model)
    cli.main(argv)


@pytest.fixture
def case(tmp_path):
    inp = tmp_path / "paciente 1"
    inp.mkdir()
    for name in ("frente.jpg", "esquerda.jpg", "direita.jpg"):
        _write_photo(inp / name)
    return inp, tmp_path / "saida"


def _eight_teeth():
    up, lo = teeth_rows(SHAPE)
    return up + lo


def _args(case, *extra):
    inp, out = case
    return ["--input-dir", str(inp), "--output-dir", str(out), *extra]


def test_main_writes_teethdreamer_structure_and_report(monkeypatch, case):
    _, out = case
    _run_main(monkeypatch, _args(case), FakeModel(_eight_teeth()))
    for arch in ("upper", "lower"):
        folder = out / f"paciente_1_{arch}"
        assert sorted(os.listdir(folder)) == ["000.png", "001.png", "002.png"]
        assert cv2.imread(str(folder / "000.png")).shape == (OUTPUT_SIZE, OUTPUT_SIZE, 3)
    assert "front: 8 dentes -> 4 superior / 4 inferior" in (out / "paciente_1_report.txt").read_text()


def test_main_sanitizes_uid(monkeypatch, case):
    _, out = case
    _run_main(monkeypatch, _args(case, "--uid", "caso #1/x"), FakeModel(_eight_teeth()))
    assert (out / "caso__1_x_upper").is_dir() and (out / "caso__1_x_lower").is_dir()


def test_main_uid_defaults_to_input_folder_name(monkeypatch, case):
    _, out = case
    _run_main(monkeypatch, _args(case), FakeModel(_eight_teeth()))
    assert (out / "paciente_1_upper").is_dir()


def test_main_exits_when_a_required_view_is_missing(monkeypatch, case, capsys):
    inp, out = case
    os.remove(inp / "direita.jpg")
    with pytest.raises(SystemExit) as exc:
        _run_main(monkeypatch, _args(case), FakeModel([]))
    assert exc.value.code == 1 and "right" in capsys.readouterr().out
    assert not out.exists()


def test_main_reports_view_without_detections(monkeypatch, case):
    _, out = case
    _run_main(monkeypatch, _args(case), FakeModel([]))
    assert "front: NENHUM DENTE DETECTADO" in (out / "paciente_1_report.txt").read_text()
    for arch in ("upper", "lower"):       # sem dentes, a imagem de saida ainda e escrita, toda preta
        assert cv2.imread(str(out / f"paciente_1_{arch}" / "000.png")).sum() == 0


def test_main_debug_flag_writes_overlays_and_rgba(monkeypatch, case):
    _, out = case
    _run_main(monkeypatch, _args(case, "--debug"), FakeModel(_eight_teeth()))
    files = os.listdir(out / "debug")
    assert sum(f.endswith("_overlay.jpg") for f in files) == 3 and sum(f.endswith("_rgba.png") for f in files) == 3


def test_main_without_debug_flag_creates_no_debug_dir(monkeypatch, case):
    _, out = case
    _run_main(monkeypatch, _args(case), FakeModel(_eight_teeth()))
    assert not (out / "debug").exists()


def test_main_named_occlusal_views_become_fourth_image(monkeypatch, case):
    inp, out = case
    _write_photo(inp / "oclusal_superior.jpg")
    _write_photo(inp / "oclusal_inferior.jpg")
    _run_main(monkeypatch, _args(case), FakeModel(_eight_teeth()))
    assert "003.png" in os.listdir(out / "paciente_1_upper") and "003.png" in os.listdir(out / "paciente_1_lower")


def test_main_numbered_occlusal_files_are_currently_ignored(monkeypatch, case):
    """Caracterizacao (a confirmar com o autor): '3.jpg'/'4.jpg' sao reconhecidos por detect_view,
    mas so nomes que comecam com 'ocl' sao processados como oclusal."""
    inp, out = case
    _write_photo(inp / "3.jpg")
    _write_photo(inp / "4.jpg")
    _run_main(monkeypatch, _args(case), FakeModel(_eight_teeth()))
    assert "003.png" not in os.listdir(out / "paciente_1_upper")
    assert "003.png" not in os.listdir(out / "paciente_1_lower")


def test_main_forwards_cli_options_to_the_model(monkeypatch, case):
    model = FakeModel(_eight_teeth())
    _run_main(monkeypatch, _args(case, "--imgsz", "960", "--conf", "0.33", "--device", "cpu"), model)
    assert len(model.calls) == 3
    assert all(c["imgsz"] == 960 and c["conf"] == DETECT_MIN_CONF and c["device"] == "cpu" for c in model.calls)


def test_main_loads_model_from_model_argument(monkeypatch, case):
    seen = []
    monkeypatch.setattr(cli, "load_yolo", lambda weights: seen.append(weights) or FakeModel([]))
    cli.main(_args(case, "--model", "meu.pt"))
    assert seen == ["meu.pt"]
