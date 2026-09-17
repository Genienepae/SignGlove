"""Exibe quais gestos faltam para cada participante antes do treinamento."""

import os
import sys

PASTA_RAIZ = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PASTA_RAIZ)

from core.cobertura_coletas import participantes_completos, resumir_cobertura


def main():
    manifesto = os.path.join(PASTA_RAIZ, 'data', 'metadata', 'coletas.jsonl')
    try:
        resumo = resumir_cobertura(manifesto)
    except ValueError as erro:
        print(f'Não foi possível verificar a cobertura: {erro}')
        return 1

    if not resumo['participantes']:
        print('Ainda não há coletas com participante registrado.')
        return 0

    print('\nCOBERTURA DE COLETAS')
    print('Participante | ' + ' | '.join(f'{gesto:>6}' for gesto in resumo['gestos']))
    for participante, dados in sorted(resumo['participantes'].items()):
        valores = ' | '.join(f'{dados.get(gesto, 0):6}' for gesto in resumo['gestos'])
        print(f'{participante:12} | {valores}')

    if participantes_completos(resumo):
        print('\nCobertura mínima pronta para avaliação por participante.')
    else:
        print('\nFaltam pelo menos dois participantes com todos os gestos registrados.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
