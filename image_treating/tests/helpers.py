import types

import numpy as np


def rect_mask(shape, y0, y1, x0, x1):
    m = np.zeros(shape, np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def teeth_rows(shape=(400, 800), upper_y=100, lower_y=300, xs=(100, 250, 400, 550), w=40, h=60):
    """Duas fileiras de dentes retangulares: (superiores, inferiores)."""
    up = [rect_mask(shape, upper_y, upper_y + h, x, x + w) for x in xs]
    lo = [rect_mask(shape, lower_y, lower_y + h, x, x + w) for x in xs]
    return up, lo


def centroid_y(mask):
    return float(np.where(mask > 0)[0].mean())


class _Tensor:
    """Imita o minimo de um tensor torch: .cpu().numpy()."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _Result:
    def __init__(self, masks, confs, classes):
        self.masks = types.SimpleNamespace(data=[_Tensor(m.astype(np.float32)) for m in masks]) if masks else None
        self.boxes = types.SimpleNamespace(conf=_Tensor(np.asarray(confs, np.float32)),
                                           cls=_Tensor(np.asarray(classes, np.float32)))


class FakeModel:
    """Substitui YOLO: devolve sempre as mascaras/confiancas/classes dadas e grava os kwargs de predict.
    Sem `classes`, tudo e dente (classe 3)."""

    def __init__(self, masks, confs=None, classes=None):
        self.masks = masks
        self.confs = confs if confs is not None else [0.9] * len(masks)
        self.classes = classes if classes is not None else [3] * len(masks)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_Result(self.masks, self.confs, self.classes)]
