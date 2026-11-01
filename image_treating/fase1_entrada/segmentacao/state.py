import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ViewState:
    view: str
    path: str
    conf: float = 0.0  # confiança base (--conf)
    img: Optional[np.ndarray] = None
    raw_masks: List[np.ndarray] = field(
        default_factory=list
    )  # saída do detector (floats)
    confs: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    defects: List[np.ndarray] = field(
        default_factory=list
    )  # cárie/cavidade/trinca (detector)
    masks: List[np.ndarray] = field(
        default_factory=list
    )  # máscaras limpas, ainda sem arcada
    mask_confs: List[float] = field(
        default_factory=list
    )  # confiança de cada máscara de `masks`
    upper: List[np.ndarray] = field(default_factory=list)
    lower: List[np.ndarray] = field(default_factory=list)
    boundary: Optional[Tuple] = None  # fronteira entre arcadas (split_arches)
    n_outliers: int = 0
    stopped: bool = False  # True: o workflow para e nada é gravado

    @property
    def name(self) -> str:
        return os.path.basename(self.path)
