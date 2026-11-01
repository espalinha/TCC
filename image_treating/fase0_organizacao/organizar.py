import json
import os
import shutil

from fase0_organizacao.nomes import normalizar_caminho_relativo, normalizar_nome
from fase0_organizacao.progresso import finalizar_progresso, garantir_pasta, mostrar_progresso


# =========================================================
# GERADOR DE NOME DESTINO COM COLISAO RESOLVIDA
# =========================================================

def nome_sem_colisao(nome_normalizado: str, usados: set, colisoes: list) -> str:
    """Devolve 'nome_normalizado' acrescentando sufixo _2, _3, ... quando o nome
    ja foi reivindicado por OUTRO item da mesma execucao (colisao real entre
    nomes normalizados), e registra o nome escolhido em 'usados'."""
    candidato = nome_normalizado
    sufixo = 2
    while candidato in usados:
        candidato = f"{nome_normalizado}_{sufixo}"
        sufixo += 1
    usados.add(candidato)
    if candidato != nome_normalizado:
        colisoes.append(f"{nome_normalizado} -> {candidato}")
    return candidato


# =========================================================
# PASTA WITH_STL (apenas pastas com .stl direto)
# =========================================================

def pasta_contem_stl(caminho: str, arquivos: list) -> bool:
    """True se ha ao menos um arquivo .stl (case-insensitive) na lista."""
    return any(arquivo.lower().endswith(".stl") for arquivo in arquivos)


def _copiar_arquivo(caminho_origem: str, caminho_destino: str, overwrite: bool,
                    silencioso: bool, processados: int, total_itens: int,
                    erros: list) -> int:
    """Copia um arquivo (sobrescrevendo se overwrite) ou pula se ja existir e
    overwrite=False. Devolve 1 se copiou, 0 caso contrario. Registra erros."""
    if os.path.exists(caminho_destino) and not overwrite:
        if not silencioso:
            mostrar_progresso(processados, total_itens, f"Pula (existe): {os.path.basename(caminho_origem)}")
        return 0
    try:
        shutil.copy2(caminho_origem, caminho_destino)
        return 1
    except OSError as exc:
        erros.append(str(caminho_origem))
        if not silencioso:
            print(f"\n[ERRO] copia falhou: {caminho_origem} -> {exc}")
        return 0


def gerar_with_stl(origem: str, destino_with_stl: str, overwrite: bool = False,
                   silencioso: bool = False) -> dict:
    """Copia, achatado na raiz de 'destino_with_stl', o conteudo COMPLETO de cada
    pasta que possui ao menos um arquivo .stl diretamente nela."""
    origem = os.path.abspath(origem)
    destino_with_stl = os.path.abspath(destino_with_stl)
    garantir_pasta(destino_with_stl)

    total_pastas = 0
    total_arquivos = 0
    colisoes = []
    erros = []
    usados_raiz = set()

    pastas_com_stl = []
    for raiz_atual, _pastas, arquivos in os.walk(origem):
        if pasta_contem_stl(raiz_atual, arquivos):
            pastas_com_stl.append((raiz_atual, arquivos))

    total_itens = sum(1 + len(arquivos) for _raiz, arquivos in pastas_com_stl)
    processados = 0

    for pasta_origem, arquivos in pastas_com_stl:
        nome_pasta = normalizar_nome(os.path.basename(pasta_origem.rstrip(os.sep)))
        nome_pasta = nome_sem_colisao(nome_pasta, usados_raiz, colisoes)
        pasta_destino = os.path.join(destino_with_stl, nome_pasta)
        garantir_pasta(pasta_destino)

        usados_arquivos = set()
        for arquivo in arquivos:
            processados += 1
            nome_arquivo = nome_sem_colisao(normalizar_nome(arquivo), usados_arquivos, colisoes)
            caminho_origem = os.path.join(pasta_origem, arquivo)
            caminho_destino = os.path.join(pasta_destino, nome_arquivo)

            total_arquivos += _copiar_arquivo(
                caminho_origem, caminho_destino, overwrite, silencioso,
                processados, total_itens, erros,
            )

            if not silencioso:
                mostrar_progresso(processados, total_itens, f"Arquivo: {arquivo}")

        total_pastas += 1

    if not silencioso:
        mostrar_progresso(total_itens, total_itens, "Concluído!")
        finalizar_progresso()

    return {
        "total_pastas": total_pastas,
        "total_arquivos": total_arquivos,
        "colisoes_resolvidas": len(colisoes),
        "colisoes": colisoes,
        "erros": erros,
    }


# =========================================================
# RELATORIO DE EXECUCAO
# =========================================================

def escrever_relatorio(caminho: str, dados: dict) -> None:
    """Grava 'dados' como JSON em 'caminho' (utf-8, indentado)."""
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)


# =========================================================
# PASTA TREATED (espelho completo normalizado)
# =========================================================

def gerar_tratado(origem: str, destino_tratado: str, overwrite: bool = False,
                  silencioso: bool = False) -> dict:
    """Copia TODO o conteudo de 'origem' para 'destino_tratado', normalizando os
    nomes de pastas e arquivos (snake_case) e preservando a hierarquia relativa."""
    origem = os.path.abspath(origem)
    destino_tratado = os.path.abspath(destino_tratado)
    garantir_pasta(destino_tratado)

    total_pastas = 0
    total_arquivos = 0
    colisoes = []
    erros = []

    items = list(os.walk(origem))
    total_itens = sum(1 + len(arquivos) for _raiz, _pastas, arquivos in items)
    processados = 0

    for raiz_atual, _pastas, arquivos in items:
        rel_path = os.path.relpath(raiz_atual, origem)
        rel_norm = normalizar_caminho_relativo(rel_path) if rel_path != "." else ""
        pasta_destino = os.path.join(destino_tratado, rel_norm) if rel_norm else destino_tratado

        usados = set()
        for pasta in _pastas:
            nome_pasta = nome_sem_colisao(normalizar_nome(pasta), usados, colisoes)
            garantir_pasta(os.path.join(pasta_destino, nome_pasta))

        if rel_norm:
            garantir_pasta(pasta_destino)
            total_pastas += 1

        for arquivo in arquivos:
            processados += 1
            nome_arquivo = nome_sem_colisao(normalizar_nome(arquivo), usados, colisoes)
            caminho_origem = os.path.join(raiz_atual, arquivo)
            caminho_destino = os.path.join(pasta_destino, nome_arquivo)

            total_arquivos += _copiar_arquivo(
                caminho_origem, caminho_destino, overwrite, silencioso,
                processados, total_itens, erros,
            )

            if not silencioso:
                mostrar_progresso(processados, total_itens, f"Arquivo: {arquivo}")

    if not silencioso:
        mostrar_progresso(total_itens, total_itens, "Concluído!")
        finalizar_progresso()

    return {
        "total_pastas": total_pastas,
        "total_arquivos": total_arquivos,
        "colisoes_resolvidas": len(colisoes),
        "colisoes": colisoes,
        "erros": erros,
    }
