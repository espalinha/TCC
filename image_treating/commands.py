import subprocess
from typing import Mapping, Optional, Protocol, Sequence


class CommandRunner(Protocol):
    def run(
        self,
        cmd: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None: ...


class SubprocessRunner:
    def run(
        self,
        cmd: Sequence[str],
        cwd: Optional[str] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        subprocess.run(list(cmd), cwd=cwd, env=env, check=True)
