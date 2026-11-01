import argparse
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# o ultralytics só é importado em load_yolo; os testes nunca precisam do modelo real.
_stub = types.ModuleType("ultralytics")
_stub.YOLO = object
sys.modules.setdefault("ultralytics", _stub)

from pipeline_config import PathLayout, PipelineConfig  # noqa: E402


class FakeRunner:
    """Grava os comandos em vez de executa-los (sem GPU, sem subprocess)."""

    def __init__(self):
        self.calls = []

    def run(self, cmd, cwd=None, env=None):
        self.calls.append((list(cmd), cwd, env))


@pytest.fixture
def runner():
    return FakeRunner()


@pytest.fixture
def config(tmp_path):
    return PipelineConfig(
        input_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        uid="p",
        teethdreamer_dir=str(tmp_path / "td"),
        model_path=str(tmp_path / "best.pt"),
    )


@pytest.fixture
def layout(config):
    return PathLayout(config)


@pytest.fixture
def skip_args():
    def make(**flags):
        base = dict(skip_teethdreamer=False, skip_remove_bg=False, skip_recon=False,
                    postprocess=False, sculpt=False, gt=None, phases=(1, 2, 3))
        return argparse.Namespace(**dict(base, **flags))
    return make
