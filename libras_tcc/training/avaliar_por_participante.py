"""Executa a avaliação do SignGlove sem misturar participantes."""

import argparse
import os
import sys

PASTA_RAIZ = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PASTA_RAIZ)

from core.avaliacao import (
    carregar_dataset_por_participante,
    avaliar_svm_por_participante,
    salvar_resultado_avaliacao,
)


def main():
    parser = argparse.ArgumentParser(description='Avalia o modelo por participante.')
    parser.add_argument('--saida', help='Arquivo JSON para salvar as métricas calculadas.')
    args = parser.parse_args()
    dados = os.path.join(PASTA_RAIZ, 'data', 'gestures')
    manifesto = os.path.join(PASTA_RAIZ, 'data', 'metadata', 'coletas.jsonl')
    try:
        x, y, grupos = carregar_dataset_por_participante(dados, manifesto)
        resultado = avaliar_svm_por_participante(x, y, grupos)
    except ValueError as erro:
        print(f'\nAvaliação por participante indisponível: {erro}')
        print('Colete os mesmos gestos com códigos como P01, P02 e P03.')
        return 1

    print('\nAVALIAÇÃO POR PARTICIPANTE')
    print(f"Participantes: {', '.join(resultado['participantes'])}")
    print(f"Gestos: {', '.join(resultado['gestos'])}")
    print(f"Amostras registradas: {resultado['amostras']}")
    print(f"Acurácia: {resultado['acuracia'] * 100:.1f}%")
    print(f"F1 macro: {resultado['f1_macro'] * 100:.1f}%")
    print('Cada divisão reserva uma pessoa inteira para teste.')
    print('\nPOR GESTO')
    print('Gesto | Precisão | Recall | F1 | Amostras')
    for gesto, metricas in resultado['por_gesto'].items():
        print(f"{gesto:5} | {metricas['precisao'] * 100:8.1f}% | "
              f"{metricas['recall'] * 100:6.1f}% | {metricas['f1'] * 100:5.1f}% | "
              f"{metricas['amostras']:8}")
    print('\nMATRIZ DE CONFUSÃO (linhas: gesto real; colunas: previsão)')
    print('      ' + ' '.join(f'{gesto:>5}' for gesto in resultado['gestos']))
    for gesto, linha in zip(resultado['gestos'], resultado['matriz_confusao']):
        print(f'{gesto:>5} ' + ' '.join(f'{valor:5}' for valor in linha))
    if args.saida:
        salvar_resultado_avaliacao(args.saida, resultado)
        print(f'\nResultado salvo em: {args.saida}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
