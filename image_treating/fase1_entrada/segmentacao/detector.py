from typing import List, NamedTuple, Optional, Protocol, Tuple

import cv2
import numpy as np

TOOTH_CLASS_ID = 3  # classes do best.pt: 0 Caries, 1 Cavity, 2 Crack, 3 Tooth
DETECT_MIN_CONF = 0.05  # o modelo devolve tudo acima disso; o corte de confiança dos dentes é uma etapa
DEFECT_MIN_CONF = 0.10  # cárie/cavidade/trinca com confiança abaixo disso é ignorada


class Detections(NamedTuple):
    masks: List[
        np.ndarray
    ]  # dentes com conf >= DETECT_MIN_CONF (float, tamanho pedido)
    confs: np.ndarray  # confiança de cada dente
    defects: List[np.ndarray]  # cárie/cavidade/trinca: viram parte do dente que tocam


class Detector(Protocol):
    def detect(self, img_path: str, shape: Tuple[int, ...]) -> Detections:
        """Máscaras no tamanho `shape[:2]`; listas vazias se nada foi detectado."""


def load_yolo(weights: str):
    from ultralytics import \
        YOLO  # import tardio: só quem roda o modelo precisa do ultralytics

    return YOLO(weights)


class YoloDetector:
    """
    Pede todas as classes e as máscaras na resolução do letterbox (retina_masks=False): tira o padding e
    redimensiona para `shape`, a resolução de trabalho. É o mesmo caminho usado para gerar as entradas
    do treino do v10 (scripts/yolo_v2/cache_yolo.py), então treino e inferência veem a mesma segmentação.
    """

    def __init__(
        self, model, imgsz: int, conf: float, device: Optional[str] = None
    ) -> None:
        self._model, self._imgsz, self._conf, self._device = model, imgsz, conf, device

    def detect(self, img_path, shape):
        kwargs = dict(
            source=img_path,
            imgsz=self._imgsz,
            conf=min(self._conf, DETECT_MIN_CONF),
            retina_masks=False,
            verbose=False,
        )
        if self._device is not None:
            kwargs["device"] = self._device
        result = self._model.predict(**kwargs)[0]
        if result.masks is None or len(result.masks.data) == 0:
            return Detections([], np.zeros(0, np.float32), [])
        orig = getattr(result, "orig_shape", None) or shape[:2]
        masks = [self._fit(m.cpu().numpy(), orig, shape) for m in result.masks.data]
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        tooth = classes == TOOTH_CLASS_ID
        defects = [
            m for m, t, c in zip(masks, tooth, confs) if not t and c >= DEFECT_MIN_CONF
        ]
        return Detections([m for m, t in zip(masks, tooth) if t], confs[tooth], defects)

    @staticmethod
    def _fit(mask, orig, shape):
        """Máscara do letterbox -> recorta o padding (proporção da foto original) -> tamanho `shape`."""
        mh, mw = mask.shape[:2]
        h, w = orig[:2]
        g = min(mh / h, mw / w)
        ph, pw = (mh - h * g) / 2, (mw - w * g) / 2
        mask = mask[
            int(round(ph - 0.1)) : int(round(mh - ph + 0.1)),
            int(round(pw - 0.1)) : int(round(mw - pw + 0.1)),
        ]
        if mask.shape[:2] == tuple(shape[:2]):
            return mask
        return cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
