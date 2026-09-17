"""Mostra um resumo das sessões de prática registradas pelo treinador."""

import os
import sys

PASTA_RAIZ = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PASTA_RAIZ)

from core.relatorio_praticas import resumir_praticas


def main():
    caminho = os.path.join(PASTA_RAIZ, 'data', 'metadata', 'praticas.jsonl')
    try:
        resumo = resumir_praticas(caminho)
    except ValueError as erro:
        print(f'Não foi possível gerar o resumo: {erro}')
        return 1

    if not resumo['sessoes']:
        print('Ainda não há sessões de desafio registradas.')
        return 0

    print('\nRESUMO DE PRÁTICAS')
    print(f"Sessões: {resumo['sessoes']}")
    print('Participante | Sessões | Acertos | Erros | Tempo')
    for codigo, dados in sorted(resumo['participantes'].items()):
        minutos, segundos = divmod(dados['duracao_segundos'], 60)
        print(f"{codigo:12} | {dados['sessoes']:8} | {dados['acertos']:7} | "
              f"{dados['erros']:5} | {dados['taxa_acerto'] * 100:4.1f}% | "
              f"{minutos}m {segundos:02}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
