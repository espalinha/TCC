import fase1_entrada.segmentacao.views as st
import pytest


@pytest.mark.parametrize(
    "filename, expected",
    [
        # convencao numerada do TeethDreamer
        ("0.jpg", st.VIEW_FRONT),
        ("1.png", st.VIEW_LEFT),
        ("2.JPG", st.VIEW_RIGHT),
        ("3.jpg", st.VIEW_UPPER_OCC),
        ("4.jpeg", st.VIEW_LOWER_OCC),
        ("5.jpg", None),
        ("10.jpg", None),
        ("01.jpg", None),
        # frente
        ("frente.jpg", st.VIEW_FRONT),
        ("Frente.JPG", st.VIEW_FRONT),
        ("FRONT_view.png", st.VIEW_FRONT),
        ("anterior.jpg", st.VIEW_FRONT),
        ("central.jpg", st.VIEW_FRONT),
        ("Anterior_1.jpg", st.VIEW_FRONT),
        # esquerda / direita
        ("esquerda.jpg", st.VIEW_LEFT),
        ("ESQUERDA.JPG", st.VIEW_LEFT),
        ("lado_left.png", st.VIEW_LEFT),
        ("direita.jpg", st.VIEW_RIGHT),
        ("Right.png", st.VIEW_RIGHT),
        ("foto_direita_2.jpg", st.VIEW_RIGHT),
        # oclusais
        ("oclusal_superior.jpg", st.VIEW_UPPER_OCC),
        ("Oclusal Superior.JPG", st.VIEW_UPPER_OCC),
        ("occlusal_upper.png", st.VIEW_UPPER_OCC),
        ("maxilar.jpg", st.VIEW_UPPER_OCC),
        ("oclusal_inferior.jpg", st.VIEW_LOWER_OCC),
        ("Oclusal_Inferior.png", st.VIEW_LOWER_OCC),
        ("occlusal_lower.png", st.VIEW_LOWER_OCC),
        ("mandibular.jpg", st.VIEW_LOWER_OCC),
        # oclusal sem indicacao de arcada: nao arrisca
        ("oclusal.jpg", None),
        ("ocl_sup.jpg", None),
        ("occlusal_foto.jpg", None),
        # fora do protocolo
        ("pan.jpg", None),
        ("panoramica.jpg", None),
        ("radiografia_panoramica.png", None),
        ("frente_no_bg.png", None),
        ("foto_no_bg.png", None),
        # nao reconhecidos
        ("xyz.jpg", None),
        ("IMG_0001.jpg", None),
        ("", None),
    ],
)
def test_detect_view(filename, expected):
    assert st.detect_view(filename) == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("frente_esquerda.jpg", st.VIEW_FRONT),  # frente tem prioridade sobre esquerda
        ("left_front.jpg", st.VIEW_FRONT),
        (
            "direita_central.jpg",
            st.VIEW_FRONT,
        ),  # 'central' vira front mesmo com 'direita'
        ("esquerda_direita.jpg", st.VIEW_LEFT),  # esquerda tem prioridade sobre direita
        (
            "oclusal_direita.jpg",
            None,
        ),  # oclusal sem superior/inferior nao cai em 'direita'
        (
            "frente_oclusal_superior.jpg",
            st.VIEW_UPPER_OCC,
        ),  # oclusal vem antes de frente
    ],
)
def test_detect_view_priority_rules(filename, expected):
    assert st.detect_view(filename) == expected


def test_detect_view_ignores_directory_part():
    assert st.detect_view("/dados/left/foto.jpg") is None
    assert st.detect_view("/dados/frente.jpg") == st.VIEW_FRONT


def test_detect_view_ignores_accents_spaces_and_extra_dots():
    assert st.detect_view("  Frénte .jpg") == st.VIEW_FRONT
    assert st.detect_view("front.right.jpg") == st.VIEW_FRONT
    assert st.detect_view("ESQUÉRDA.jpg") == st.VIEW_LEFT


def test_normalize_strips_accents_and_lowercases():
    assert st._normalize("ÂNTERIOR Éé") == "anterior ee"


def test_find_views_maps_each_view_once(tmp_path, capsys):
    for name in ("frente.jpg", "esquerda.png", "direita.jpeg"):
        (tmp_path / name).write_bytes(b"x")
    views = st.find_views(str(tmp_path))
    assert set(views) == {st.VIEW_FRONT, st.VIEW_LEFT, st.VIEW_RIGHT}
    assert views[st.VIEW_FRONT].endswith("frente.jpg")
    assert capsys.readouterr().out == ""


def test_find_views_skips_unknown_duplicates_and_non_images(tmp_path, capsys):
    (tmp_path / "frente.jpg").write_bytes(b"x")
    (tmp_path / "front_2.jpg").write_bytes(
        b"x"
    )  # duplicada: ordem alfabetica, a primeira vence
    (tmp_path / "xyz.jpg").write_bytes(b"x")  # view nao reconhecida
    (tmp_path / "esquerda.txt").write_bytes(
        b"x"
    )  # extensao invalida: ignorada em silencio
    (tmp_path / "ESQUERDA.JPG").write_bytes(b"x")  # extensao em maiuscula vale
    (tmp_path / "direita").mkdir()  # diretorio nao conta
    views = st.find_views(str(tmp_path))
    out = capsys.readouterr().out
    assert views[st.VIEW_FRONT].endswith("frente.jpg")
    assert views[st.VIEW_LEFT].endswith("ESQUERDA.JPG")
    assert st.VIEW_RIGHT not in views
    assert "xyz.jpg" in out and "front_2.jpg" in out and "esquerda.txt" not in out


def test_find_views_ignores_subdirectories_content(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "frente.jpg").write_bytes(b"x")
    assert st.find_views(str(tmp_path)) == {}


def test_find_views_empty_dir_and_missing_dir(tmp_path):
    assert st.find_views(str(tmp_path)) == {}
    with pytest.raises(FileNotFoundError):
        st.find_views(str(tmp_path / "nao_existe"))


@pytest.mark.parametrize(
    "name, expected",
    [
        ("oclusal_superior.jpg", True),
        ("Oclusão inferior.JPG", True),
        ("3.jpg", False),
        ("superior.jpg", False),
    ],
)
def test_is_named_occlusal_only_accepts_names_starting_with_ocl(name, expected):
    assert st.is_named_occlusal("/x/" + name) is expected
