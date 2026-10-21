"""Contrato comum de todas as etapas do pipeline."""

from typing import Protocol


class Step(Protocol):
    @property
    def name(self) -> str:
        ...

    def run(self) -> None:
        ...
