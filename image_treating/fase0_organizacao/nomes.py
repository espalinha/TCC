import os
import re
import unicodedata


def remover_acentos(texto: str) -> str:
    """Remove acentos/diacríticos de uma string."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalizar_nome(nome: str) -> str:
    """
    Normaliza um nome de arquivo ou pasta:
    - separa a extensão (se houver) para deixá-la minúscula sem mexer mais nela
    - remove acentos
    - converte para minúsculo
    - troca espaços (e sequências de espaços) por "_"
    - remove caracteres especiais indesejados
    - remove underscores duplicados/nas pontas
    """
    base, ext = os.path.splitext(nome)

    base = remover_acentos(base)
    base = base.lower()

    base = re.sub(r"\s+", "_", base.strip())
    base = re.sub(r"[^a-z0-9_\-.]", "", base)
    base = re.sub(r"_+", "_", base).strip("_")

    ext = remover_acentos(ext).lower()
    ext = re.sub(r"\s+", "_", ext)

    if not base:
        base = "sem_nome"

    return base + ext


def normalizar_caminho_relativo(rel_path: str) -> str:
    """
    Normaliza cada parte de um caminho relativo (ex: "Pasta A/Sub Pasta")
    devolvendo, por exemplo, "pasta_a/sub_pasta".
    """
    if rel_path in (".", ""):
        return ""
    partes = rel_path.split(os.sep)
    partes_normalizadas = [normalizar_nome(p) for p in partes]
    return os.path.join(*partes_normalizadas)
