"""Mostra o estado do SignGlove antes de uma coleta, treino ou demonstração."""

import os
import sys

PASTA_RAIZ = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PASTA_RAIZ)

from core.diagnostico import diagnosticar_projeto


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Mostra o estado do SignGlove.')
    parser.add_argument(
        '--strict', action='store_true',
        help='retorna código 1 se os dados não estiverem prontos para treino',
    )
    args = parser.parse_args()
    resultado = diagnosticar_projeto(
        os.path.join(PASTA_RAIZ, 'data', 'gestures'),
        os.path.join(PASTA_RAIZ, 'data', 'metadata', 'coletas.jsonl'),
        os.path.join(PASTA_RAIZ, 'models', 'modelo_libras.pkl'),
    )
    print('\nDIAGNÓSTICO DO SIGNGLOVE')
    print(f"Gestos: {len(resultado['amostras_por_gesto'])}")
    for gesto, quantidade in resultado['amostras_por_gesto'].items():
        print(f'  {gesto}: {quantidade} amostras')
    print(f"Total: {resultado['total_amostras']} amostras")
    print('Dimensões dos vetores: ' + (', '.join(map(str, resultado['dimensoes_features'])) or 'nenhuma'))
    print(f"Amostras inválidas: {resultado['amostras_invalidas']}")
    print('Modelo salvo: ' + ('sim' if resultado['modelo_encontrado'] else 'não'))
    print('Pronto para treino: ' + ('sim' if resultado['pronto_treino'] else 'não'))
    print('Avaliação por participante: ' + (
        'pronta' if resultado['avaliacao_por_participante_pronta'] else 'faltam coletas'))
    if args.strict and not resultado['pronto_treino']:
        print('Diagnóstico estrito: dados ainda não estão prontos para treino.')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
