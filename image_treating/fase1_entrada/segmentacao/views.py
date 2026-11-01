import os
import unicodedata

VIEW_FRONT = "front"
VIEW_LEFT = "left"
VIEW_RIGHT = "right"
VIEW_UPPER_OCC = "upper_occlusal"
VIEW_LOWER_OCC = "lower_occlusal"

REQUIRED_VIEWS = [VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT]
IMAGE_EXTS = (".jpg", ".jpeg", ".png")

# convenção numerada 0..4 do TeethDreamer
_NUMBERED = {
    "0": VIEW_FRONT,
    "1": VIEW_LEFT,
    "2": VIEW_RIGHT,
    "3": VIEW_UPPER_OCC,
    "4": VIEW_LOWER_OCC,
}
_OCCLUSAL_WORDS = ("oclusal", "occlusal", "maxilar", "mandibular")
_UPPER_WORDS = ("superior", "maxilar", "upper")
_LOWER_WORDS = ("inferior", "mandibular", "lower")
# ordem importa: frente vence esquerda, que vence direita
_LATERAL_WORDS = (
    (VIEW_FRONT, ("frente", "front", "anterior", "central")),
    (VIEW_LEFT, ("esquerda", "left")),
    (VIEW_RIGHT, ("direita", "right")),
)


def _normalize(text):
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _contains_any(stem, words):
    return any(w in stem for w in words)


def _occlusal_view(stem):
    """Oclusal sem indicação de arcada devolve None: não arriscar."""
    if _contains_any(stem, _UPPER_WORDS):
        return VIEW_UPPER_OCC
    if _contains_any(stem, _LOWER_WORDS):
        return VIEW_LOWER_OCC
    return None


def detect_view(filename):
    """Retorna a view da foto pelo nome do arquivo, ou None se não reconhecida."""
    stem = _normalize(os.path.splitext(os.path.basename(filename))[0]).strip()

    if "no_bg" in stem:
        return None
    if stem in _NUMBERED:
        return _NUMBERED[stem]
    if stem == "pan" or "panoram" in stem:
        return None  # radiografia panorâmica: fora do protocolo de 5 fotos
    if stem.startswith("ocl") or _contains_any(stem, _OCCLUSAL_WORDS):
        return _occlusal_view(stem)
    for view, words in _LATERAL_WORDS:
        if _contains_any(stem, words):
            return view
    return None


def is_named_occlusal(path):
    """True se o NOME do arquivo começa com "ocl" (oclusais numeradas, 3.jpg/4.jpg, são ignoradas de propósito)."""
    return _normalize(os.path.splitext(os.path.basename(path))[0]).startswith("ocl")


def find_views(input_dir):
    """Mapeia view -> caminho do arquivo, varrendo só o nível superior da pasta."""
    views = {}
    for fname in sorted(os.listdir(input_dir)):
        fpath = os.path.join(input_dir, fname)
        if not os.path.isfile(fpath) or not fname.lower().endswith(IMAGE_EXTS):
            continue
        view = detect_view(fname)
        if view is None:
            print(f"  [skip] {fname} (view não reconhecida)")
            continue
        if view in views:
            print(
                f"  [skip] {fname} (view '{view}' já mapeada para {os.path.basename(views[view])})"
            )
            continue
        views[view] = fpath
    return views
