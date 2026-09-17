"""Resumo de cobertura de gestos por participante no manifesto de coleta."""

from __future__ import annotations

import json
from pathlib import Path


def resumir_cobertura(arquivo_manifesto: str | Path) -> dict:
    caminho = Path(arquivo_manifesto)
    if not caminho.exists():
        return {'participantes': {}, 'gestos': []}

    participantes = {}
    gestos = set()
    for numero_linha, linha in enumerate(caminho.read_text(encoding='utf-8').splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            lote = json.loads(linha)
            participante = lote['participante']
            gesto = lote['gesto']
            quantidade = int(lote['quantidade'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as erro:
            raise ValueError(f'Metadado inválido na linha {numero_linha}: {erro}') from erro
        if quantidade < 1:
            raise ValueError(f'Metadado inválido na linha {numero_linha}: quantidade deve ser positiva.')
        gestos.add(gesto)
        por_gesto = participantes.setdefault(participante, {})
        por_gesto[gesto] = por_gesto.get(gesto, 0) + quantidade

    return {'participantes': participantes, 'gestos': sorted(gestos)}


def participantes_completos(resumo: dict) -> bool:
    gestos = resumo['gestos']
    participantes = resumo['participantes']
    return (len(participantes) >= 2 and len(gestos) >= 2 and
            all(all(dados.get(gesto, 0) > 0 for gesto in gestos)
                for dados in participantes.values()))
