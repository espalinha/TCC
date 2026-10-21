"""Excecoes do pipeline."""


class PipelineError(Exception):
    """Falha em uma etapa do pipeline; interrompe a execucao."""
