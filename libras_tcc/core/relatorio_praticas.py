"""Resumo agregando sessões anônimas do modo desafio."""

from __future__ import annotations

import json
from pathlib import Path


def resumir_praticas(arquivo: str | Path) -> dict:
    caminho = Path(arquivo)
    if not caminho.exists():
        return {'sessoes': 0, 'participantes': {}}

    participantes = {}
    for numero_linha, linha in enumerate(caminho.read_text(encoding='utf-8').splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            registro = json.loads(linha)
            codigo = registro['participante']
            acertos = int(registro['acertos'])
            erros = int(registro.get('erros', 0))
            duracao = int(registro['duracao_segundos'])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as erro:
            raise ValueError(f'Registro inválido na linha {numero_linha}: {erro}') from erro
        if acertos < 0 or erros < 0 or duracao < 0:
            raise ValueError(f'Registro inválido na linha {numero_linha}: valores negativos.')

        total = participantes.setdefault(codigo, {'sessoes': 0, 'acertos': 0, 'erros': 0, 'duracao_segundos': 0})
        total['sessoes'] += 1
        total['acertos'] += acertos
        total['erros'] += erros
        total['duracao_segundos'] += duracao

    for total in participantes.values():
        tentativas = total['acertos'] + total['erros']
        total['taxa_acerto'] = (total['acertos'] / tentativas) if tentativas else 0.0

    return {'sessoes': sum(item['sessoes'] for item in participantes.values()), 'participantes': participantes}
