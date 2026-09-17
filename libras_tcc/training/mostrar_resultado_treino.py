"""Exibe um resumo JSON de treinamento sem iniciar câmera ou interface."""

import argparse
import json
import sys
from pathlib import Path

PASTA_RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PASTA_RAIZ))

from core.relatorio_treino import formatar_resumo_treino


def main():
    parser = argparse.ArgumentParser(description='Exibe um resumo JSON de treinamento.')
    parser.add_argument('arquivo', help='caminho do JSON criado por treinar_modelo.py --saida')
    args = parser.parse_args()
    try:
        resumo = json.loads(Path(args.arquivo).read_text(encoding='utf-8'))
        print(formatar_resumo_treino(resumo))
    except (OSError, json.JSONDecodeError, ValueError, TypeError, KeyError) as erro:
        print(f'[ERRO] Não foi possível abrir o resumo: {erro}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
