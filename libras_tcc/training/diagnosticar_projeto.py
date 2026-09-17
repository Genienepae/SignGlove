"""Mostra o estado do SignGlove antes de uma coleta, treino ou demonstração."""

import os
import sys

PASTA_RAIZ = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PASTA_RAIZ)

from core.diagnostico import diagnosticar_projeto


def main():
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
    print('Modelo salvo: ' + ('sim' if resultado['modelo_encontrado'] else 'não'))
    print('Pronto para treino: ' + ('sim' if resultado['pronto_treino'] else 'não'))
    print('Avaliação por participante: ' + (
        'pronta' if resultado['avaliacao_por_participante_pronta'] else 'faltam coletas'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
