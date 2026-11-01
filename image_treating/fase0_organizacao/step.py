import os

from errors import PipelineError
from fase0_organizacao.organizar import escrever_relatorio, gerar_tratado, gerar_with_stl
from pipeline_config import PathLayout, PipelineConfig


class OrganizeAcervoStep:
    name = "Organização do acervo (fase 0)"

    def __init__(self, config: PipelineConfig, layout: PathLayout) -> None:
        self._config, self._layout = config, layout

    def run(self) -> None:
        raw = self._config.raw_dir
        if not raw or not os.path.isdir(raw):
            raise PipelineError(f"{self.name}: pasta do acervo bruto não encontrada ({raw}); use --raw-dir")
        overwrite = self._config.organize_overwrite
        l = self._layout
        jobs = [
            ("treated", l.treated_dir, gerar_tratado, "report_treated.json"),
            ("with_stl", l.with_stl_dir, gerar_with_stl, "report_with_stl.json"),
        ]
        errors = []
        for label, dest, build, report in jobs:
            print(f"  {label}/ -> {dest}")
            stats = build(raw, dest, overwrite=overwrite)
            escrever_relatorio(os.path.join(dest, report), {"origem": raw, "destino": dest, **stats})
            print(f"  {label}: {stats['total_arquivos']} arquivos, {stats['total_pastas']} pastas, "
                  f"{stats['colisoes_resolvidas']} colisões, {len(stats['erros'])} erros")
            errors += stats["erros"]
        if errors:
            raise PipelineError(f"{self.name}: {len(errors)} arquivo(s) não copiado(s); veja os relatórios")
