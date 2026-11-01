import argparse
import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SegmentationConfig:
    input_dir: str
    output_dir: str
    uid: str
    model: str = "best.pt"
    imgsz: int = 1920
    conf: float = 0.10
    device: Optional[str] = None
    debug: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "SegmentationConfig":
        uid = args.uid or os.path.basename(
            os.path.abspath(args.input_dir).rstrip(os.sep)
        )
        uid = "".join(c if (c.isalnum() or c in "-_") else "_" for c in uid)
        return cls(
            args.input_dir,
            args.output_dir,
            uid,
            args.model,
            args.imgsz,
            args.conf,
            args.device,
            args.debug,
        )
