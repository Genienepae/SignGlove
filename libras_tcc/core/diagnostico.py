"""Diagnóstico local dos arquivos necessários para uma demonstração do projeto."""

from __future__ import annotations

import json
from pathlib import Path

from .cobertura_coletas import participantes_completos, resumir_cobertura


def diagnosticar_projeto(pasta_dados: str | Path, manifesto: str | Path, modelo: str | Path) -> dict:
    pasta_dados = Path(pasta_dados)
    amostras_por_gesto = {}
    if pasta_dados.exists():
        for arquivo in sorted(pasta_dados.glob('*.json')):
            try:
                dados = json.loads(arquivo.read_text(encoding='utf-8'))
            except json.JSONDecodeError as erro:
                raise ValueError(f'Arquivo de dados inválido: {arquivo.name}: {erro}') from erro
            if not isinstance(dados, list):
                raise ValueError(f'Arquivo de dados inválido: {arquivo.name} não contém uma lista.')
            amostras_por_gesto[arquivo.stem] = len(dados)

    cobertura = resumir_cobertura(manifesto)
    pronto_treino = len(amostras_por_gesto) >= 2 and min(amostras_por_gesto.values(), default=0) >= 20
    return {
        'amostras_por_gesto': amostras_por_gesto,
        'total_amostras': sum(amostras_por_gesto.values()),
        'modelo_encontrado': Path(modelo).exists(),
        'pronto_treino': pronto_treino,
        'avaliacao_por_participante_pronta': participantes_completos(cobertura),
        'cobertura': cobertura,
    }
