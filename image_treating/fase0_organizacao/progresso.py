import os
import sys


def mostrar_progresso(atual: int, total: int, mensagem: str, largura: int = 30):
    """Imprime uma barra de progresso na mesma linha do terminal usando \\r."""
    if total <= 0:
        percentual = 100.0
    else:
        percentual = (atual / total) * 100

    preenchido = int(largura * percentual / 100)
    barra = "#" * preenchido + "-" * (largura - preenchido)

    # limita o tamanho da mensagem para não quebrar a linha no terminal
    msg_curta = (mensagem[:60] + "...") if len(mensagem) > 63 else mensagem

    linha = f"\r[{barra}] {percentual:6.2f}% | {msg_curta}"
    sys.stdout.write(linha + " " * 10)  # espaços extras apagam sobras de linhas anteriores
    sys.stdout.flush()


def finalizar_progresso():
    """Pula uma linha ao final da barra de progresso (chamar depois do loop)."""
    print()


def garantir_pasta(caminho: str):
    """Cria a pasta (e pais) se não existir. Não dá erro se já existir."""
    os.makedirs(caminho, exist_ok=True)
