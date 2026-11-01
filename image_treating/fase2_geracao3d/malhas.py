import os
import re
from glob import glob
from typing import Optional

ARCHES = ("upper", "lower")


def arch_of(name: str) -> str:
    """'lower' ou 'upper' a partir do nome do arquivo."""
    return "lower" if "lower" in name else "upper"


def find_final_obj(arch_dir: str) -> Optional[str]:
    """.obj de maior iteração em <arch_dir>/*/neus/save/*.obj."""
    def rank(path: str):
        m = re.match(r"it(\d+)", os.path.basename(path))
        return (int(m.group(1)) if m else 0, os.path.getmtime(path))

    return max(glob(os.path.join(arch_dir, "*", "neus", "save", "*.obj")), key=rank, default=None)
