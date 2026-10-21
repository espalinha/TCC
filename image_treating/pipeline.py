"""Orquestrador: executa uma lista de Steps em ordem."""

from typing import Sequence

from steps.base import Step


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        raise NotImplementedError

    def run(self) -> None:
        raise NotImplementedError
