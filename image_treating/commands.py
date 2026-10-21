"""Execucao de comandos externos atras de uma abstracao (permite testar sem GPU)."""

from typing import Mapping, Optional, Protocol, Sequence


class CommandRunner(Protocol):
    def run(
        self,
        cmd: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        ...


class SubprocessRunner:
    """Implementacao real: delega para subprocess.run(check=True)."""

    def run(
        self,
        cmd: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        raise NotImplementedError
