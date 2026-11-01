import os

from fase0_organizacao.organizar import (
    gerar_tratado,
    gerar_with_stl,
    nome_sem_colisao,
    pasta_contem_stl,
)


def _escrever(path, conteudo="x"):
    with open(path, "w") as f:
        f.write(conteudo)


def _listar_arquivos(raiz):
    saida = []
    for pasta, _, arqs in os.walk(raiz):
        for a in arqs:
            saida.append(os.path.relpath(os.path.join(pasta, a), raiz))
    return sorted(saida)


# ---- R2: normalizacao de nomes (via nome_sem_colisao + normalizar_nome) ----

def test_normalizacao_espacos_acentos_maiusculas(tmp_path):
    origem = tmp_path / "origem"
    pasta = origem / "Paciente A" / "Sub Pasta"
    pasta.mkdir(parents=True)
    _escrever(pasta / "Laudo Pan 100480.JPG")
    _escrever(pasta / "superior.STL")

    gerar_tratado(str(origem), str(tmp_path / "treated"), silencioso=True)

    assert _listar_arquivos(tmp_path / "treated") == [
        os.path.join("paciente_a", "sub_pasta", "laudo_pan_100480.jpg"),
        os.path.join("paciente_a", "sub_pasta", "superior.stl"),
    ]


def test_normalizacao_nome_vazio(tmp_path):
    origem = tmp_path / "origem"
    pasta = origem / "Paciente"
    pasta.mkdir(parents=True)
    _escrever(pasta / "$$$")  # sem caracteres validos -> sem_nome

    gerar_tratado(str(origem), str(tmp_path / "treated"), silencioso=True)

    arquivos = _listar_arquivos(tmp_path / "treated")
    assert os.path.join("paciente", "sem_nome") in arquivos


# ---- R3: treated espelho completo normalizado ----

def test_gerar_tratado_espelha_tudo(tmp_path):
    origem = tmp_path / "origem"
    (origem / "i1" / "Paciente").mkdir(parents=True)
    _escrever(origem / "i1" / "Paciente" / "central.jpg")
    _escrever(origem / "i1" / "Paciente" / "superior.stl")
    _escrever(origem / "raiz.txt")

    stats = gerar_tratado(str(origem), str(tmp_path / "treated"), silencioso=True)

    assert stats["total_arquivos"] == 3
    assert _listar_arquivos(tmp_path / "treated") == [
        "i1/paciente/central.jpg",
        "i1/paciente/superior.stl",
        "raiz.txt",
    ]


# ---- R4: with_stl achatado com conteudo completo ----

def test_gerar_with_stl_achata(tmp_path):
    origem = tmp_path / "origem"
    (origem / "imagens_1" / "Paciente A").mkdir(parents=True)
    (origem / "imagens_2" / "Paciente B").mkdir(parents=True)
    _escrever(origem / "imagens_1" / "Paciente A" / "superior.STL")
    _escrever(origem / "imagens_1" / "Paciente A" / "central.jpg")
    _escrever(origem / "imagens_2" / "Paciente B" / "inferior.stl")

    stats = gerar_with_stl(str(origem), str(tmp_path / "with_stl"), silencioso=True)

    assert stats["total_pastas"] == 2
    assert _listar_arquivos(tmp_path / "with_stl") == [
        "paciente_a/central.jpg",
        "paciente_a/superior.stl",
        "paciente_b/inferior.stl",
    ]


# ---- R5: pasta sem .stl direto ignorada ----

def test_with_stl_ignora_pasta_sem_stl_direto(tmp_path):
    origem = tmp_path / "origem"
    (origem / "imagens_1" / "Paciente A").mkdir(parents=True)
    (origem / "imagens_1" / "Paciente B" / "sub").mkdir(parents=True)
    _escrever(origem / "imagens_1" / "Paciente A" / "superior.stl")
    # .stl apenas em subnivel -> pasta "Paciente B" NAO conta (mas "sub" conta)
    _escrever(origem / "imagens_1" / "Paciente B" / "sub" / "inferior.stl")

    stats = gerar_with_stl(str(origem), str(tmp_path / "with_stl"), silencioso=True)

    assert stats["total_pastas"] == 2
    assert sorted(os.listdir(tmp_path / "with_stl")) == ["paciente_a", "sub"]


def test_pasta_contem_stl_case_insensitive():
    assert pasta_contem_stl("/x", ["a.jpg", "B.STL"])
    assert pasta_contem_stl("/x", ["a.jpg", "b.stl"])
    assert not pasta_contem_stl("/x", ["a.jpg", "b.ply"])


# ---- R6: nao sobrescreve por padrao / sobrescreve com overwrite ----

def test_sem_overwrite_nao_sobrescreve(tmp_path):
    origem = tmp_path / "origem"
    (origem / "p").mkdir(parents=True)
    _escrever(origem / "p" / "a.txt", "novo")
    destino = tmp_path / "treated"
    (destino / "p").mkdir(parents=True)
    _escrever(destino / "p" / "a.txt", "velho")

    gerar_tratado(str(origem), str(destino), silencioso=True)

    assert (destino / "p" / "a.txt").read_text() == "velho"


def test_com_overwrite_sobrescreve(tmp_path):
    origem = tmp_path / "origem"
    (origem / "p").mkdir(parents=True)
    _escrever(origem / "p" / "a.txt", "novo")
    destino = tmp_path / "treated"
    (destino / "p").mkdir(parents=True)
    _escrever(destino / "p" / "a.txt", "velho")

    gerar_tratado(str(origem), str(destino), overwrite=True, silencioso=True)

    assert (destino / "p" / "a.txt").read_text() == "novo"


# ---- R4: colisao gera sufixo _2 ----

def test_colisao_gera_sufixo(tmp_path):
    origem = tmp_path / "origem"
    (origem / "i1" / "Paciente A").mkdir(parents=True)
    (origem / "i2" / "Paciente A").mkdir(parents=True)
    _escrever(origem / "i1" / "Paciente A" / "a.stl")
    _escrever(origem / "i2" / "Paciente A" / "b.stl")

    stats = gerar_with_stl(str(origem), str(tmp_path / "with_stl"), silencioso=True)

    assert sorted(os.listdir(tmp_path / "with_stl")) == ["paciente_a", "paciente_a_2"]
    assert stats["colisoes_resolvidas"] == 1


def test_nome_sem_colisao_registra_usados():
    colisoes = []
    usados = set()
    primeiro = nome_sem_colisao("paciente", usados, colisoes)
    segundo = nome_sem_colisao("paciente", usados, colisoes)
    assert primeiro == "paciente"
    assert segundo == "paciente_2"
    assert colisoes == ["paciente -> paciente_2"]


# ---- R8: origem intocada ----

def test_origem_intocada(tmp_path):
    origem = tmp_path / "origem"
    (origem / "i1" / "Paciente A").mkdir(parents=True)
    _escrever(origem / "i1" / "Paciente A" / "superior.STL")

    antes = sorted(
        os.path.relpath(os.path.join(r, a), origem)
        for r, _, arqs in os.walk(origem)
        for a in arqs
    )

    gerar_tratado(str(origem), str(tmp_path / "treated"), silencioso=True)
    gerar_with_stl(str(origem), str(tmp_path / "with_stl"), silencioso=True)

    depois = sorted(
        os.path.relpath(os.path.join(r, a), origem)
        for r, _, arqs in os.walk(origem)
        for a in arqs
    )

    assert antes == depois == ["i1/Paciente A/superior.STL"]