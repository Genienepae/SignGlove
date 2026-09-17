"""Metadados de lotes de coleta para avaliações sem misturar participantes."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re


PADRAO_PARTICIPANTE = re.compile(r"^[A-Z0-9_-]{2,20}$")
PADRAO_SESSAO = re.compile(r"^[A-Z0-9_-]{2,20}$")


def normalizar_codigo_participante(codigo: str) -> str:
    """Aceita somente códigos curtos, sem nomes ou outros dados pessoais."""
    codigo = codigo.strip().upper()
    if not PADRAO_PARTICIPANTE.fullmatch(codigo):
        raise ValueError(
            'Use um código anônimo de 2 a 20 caracteres, por exemplo P01 ou ALUNO_02.')
    return codigo


def normalizar_codigo_sessao(codigo: str) -> str:
    """Aceita um código curto para agrupar gravações da mesma sessão."""
    codigo = codigo.strip().upper()
    if not PADRAO_SESSAO.fullmatch(codigo):
        raise ValueError('Use uma sessão de 2 a 20 caracteres, por exemplo S01 ou CASA_NOITE.')
    return codigo


def registrar_lote(
    arquivo_manifesto: str | Path,
    participante: str,
    gesto: str,
    arquivo_amostras: str,
    inicio: int,
    quantidade: int,
    sessao: str = 'S01',
    inicio_coleta: str | None = None,
) -> dict:
    """Adiciona uma linha JSON para um lote contínuo de amostras."""
    if quantidade < 1 or inicio < 0:
        raise ValueError('O lote precisa ter pelo menos uma amostra e início válido.')

    participante = normalizar_codigo_participante(participante)
    sessao = normalizar_codigo_sessao(sessao)
    registro = {
        'versao': 1,
        'participante': participante,
        'sessao': sessao,
        'gesto': gesto,
        'arquivo_amostras': arquivo_amostras,
        'indice_inicio': inicio,
        'indice_fim': inicio + quantidade - 1,
        'quantidade': quantidade,
        'inicio_coleta': inicio_coleta or datetime.now(timezone.utc).isoformat(),
        'registrado_em': datetime.now(timezone.utc).isoformat(),
    }
    caminho = Path(arquivo_manifesto)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open('a', encoding='utf-8') as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + '\n')
    return registro


def remover_registros_do_gesto(arquivo_manifesto: str | Path, arquivo_amostras: str) -> None:
    """Remove metadados quando todas as amostras de um gesto são apagadas."""
    caminho = Path(arquivo_manifesto)
    if not caminho.exists():
        return
    registros = []
    for linha in caminho.read_text(encoding='utf-8').splitlines():
        if not linha.strip():
            continue
        registro = json.loads(linha)
        if registro.get('arquivo_amostras') != arquivo_amostras:
            registros.append(registro)
    conteudo = ''.join(json.dumps(registro, ensure_ascii=False) + '\n' for registro in registros)
    caminho.write_text(conteudo, encoding='utf-8')
