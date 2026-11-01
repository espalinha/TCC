from typing import Sequence

from step import Step


class Pipeline:
    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)

    def run(self) -> None:
        for i, step in enumerate(self._steps, 1):
            print(f">>> STEP {i}: {step.name}")
            step.run()
            print(f"<<< STEP {i} concluido\n")
