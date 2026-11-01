import numpy as np
from PIL import Image

from fase2_geracao3d.fundo_branco import background_mask, remove_bg


def _image():
    arr = np.full((20, 20, 3), 255, np.uint8)
    arr[5:15, 5:15] = 100          # dente escuro
    arr[8:12, 8:12] = 255          # branco dentro do dente: não é fundo
    return arr


def test_background_mask_only_takes_white_connected_to_corners():
    mask = background_mask(_image())
    assert mask[0, 0] and mask[19, 19] and not mask[10, 10] and not mask[6, 6]


def test_remove_bg_makes_background_transparent(tmp_path):
    src, dst = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(_image()).save(src)
    remove_bg(str(src), str(dst))
    out = np.array(Image.open(dst))
    assert out[0, 0, 3] == 0 and out[10, 10, 3] == 255 and out[6, 6, 3] == 255
