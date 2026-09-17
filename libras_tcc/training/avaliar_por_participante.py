"""Executa a avaliação do SignGlove sem misturar participantes."""

import os
import sys

PASTA_RAIZ = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PASTA_RAIZ)

from core.avaliacao import carregar_dataset_por_participante, avaliar_svm_por_participante


def main():
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
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
