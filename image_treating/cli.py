"""Interface de linha de comando: le argumentos, monta a lista de Steps e executa."""

import argparse
from typing import List

from commands import CommandRunner
from pipeline_config import PathLayout, PipelineConfig
from steps.base import Step


def build_parser() -> argparse.ArgumentParser:
    raise NotImplementedError


def select_steps(
    args: argparse.Namespace,
    config: PipelineConfig,
    layout: PathLayout,
    runner: CommandRunner,
) -> List[Step]:
    """Aplica as regras de --skip-teethdreamer / --skip-remove-bg / --skip-recon."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError
