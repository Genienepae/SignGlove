"""Diagnóstico local dos arquivos necessários para uma demonstração do projeto."""

from __future__ import annotations

import json
import math
import numbers
from pathlib import Path

from .cobertura_coletas import participantes_completos, resumir_cobertura


def diagnosticar_projeto(pasta_dados: str | Path, manifesto: str | Path, modelo: str | Path) -> dict:
    pasta_dados = Path(pasta_dados)
    amostras_por_gesto = {}
    dimensoes_features = set()
    amostras_invalidas = 0
    if pasta_dados.exists():
        for arquivo in sorted(pasta_dados.glob('*.json')):
            try:
                dados = json.loads(arquivo.read_text(encoding='utf-8'))
            except json.JSONDecodeError as erro:
                raise ValueError(f'Arquivo de dados inválido: {arquivo.name}: {erro}') from erro
            if not isinstance(dados, list):
                raise ValueError(f'Arquivo de dados inválido: {arquivo.name} não contém uma lista.')
            amostras_por_gesto[arquivo.stem] = len(dados)
            for amostra in dados:
                if not isinstance(amostra, (list, tuple)) or not amostra:
                    amostras_invalidas += 1
                    continue
                dimensoes_features.add(len(amostra))
                if not all(isinstance(valor, numbers.Real) and math.isfinite(float(valor))
                           for valor in amostra):
                    amostras_invalidas += 1

    cobertura = resumir_cobertura(manifesto)
    pronto_treino = (
        len(amostras_por_gesto) >= 2
        and min(amostras_por_gesto.values(), default=0) >= 20
        and amostras_invalidas == 0
        and len(dimensoes_features) <= 1
    )
    return {
        'amostras_por_gesto': amostras_por_gesto,
        'total_amostras': sum(amostras_por_gesto.values()),
        'modelo_encontrado': Path(modelo).exists(),
        'pronto_treino': pronto_treino,
        'dimensoes_features': sorted(dimensoes_features),
        'amostras_invalidas': amostras_invalidas,
        'avaliacao_por_participante_pronta': participantes_completos(cobertura),
        'cobertura': cobertura,
    }
