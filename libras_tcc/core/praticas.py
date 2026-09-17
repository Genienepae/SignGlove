"""Registro anônimo de sessões de prática do modo desafio."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

from .coletas import normalizar_codigo_participante


def registrar_pratica(
    arquivo: str | Path, participante: str, inicio: str, acertos: int, erros: int = 0
) -> dict:
    """Acrescenta uma sessão de prática sem registrar imagem ou nome da pessoa."""
    if acertos < 0 or erros < 0:
        raise ValueError('Acertos e erros não podem ser negativos.')
    participante = normalizar_codigo_participante(participante)
    fim = datetime.now(timezone.utc)
    inicio_data = datetime.fromisoformat(inicio)
    duracao_segundos = max(0, round((fim - inicio_data).total_seconds()))
    registro = {
        'versao': 1,
        'participante': participante,
        'inicio': inicio_data.isoformat(),
        'fim': fim.isoformat(),
        'duracao_segundos': duracao_segundos,
        'acertos': acertos,
        'erros': erros,
    }
    caminho = Path(arquivo)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open('a', encoding='utf-8') as destino:
        destino.write(json.dumps(registro, ensure_ascii=False) + '\n')
    return registro
