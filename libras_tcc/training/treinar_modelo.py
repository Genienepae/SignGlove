"""
training/treinar_modelo.py
--------------------------
Carrega todos os arquivos de dados coletados e treina o modelo.

Como usar:
    python training/treinar_modelo.py

O modelo treinado é salvo em: models/modelo_libras.pkl
"""

import os
import sys
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.classifier import ClassificadorGestos
from core.avaliacao import carregar_dataset_por_participante, avaliar_svm_por_participante

# Sklearn para métricas de avaliação
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def salvar_resumo_treino(
    caminho, acuracia, f1_macro, criterio_metrica, classes, amostras, por_gesto,
    matriz_confusao
):
    """Salva as métricas principais do treinamento em JSON."""
    destino = os.path.abspath(caminho)
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    with open(destino, 'w', encoding='utf-8') as arquivo:
        json.dump({
            'acuracia': float(acuracia),
            'f1_macro': float(f1_macro),
            'criterio_metrica': criterio_metrica,
            'gestos': sorted(set(classes)),
            'amostras': int(amostras),
            'por_gesto': por_gesto,
            'matriz_confusao': matriz_confusao,
        }, arquivo, ensure_ascii=False, indent=2)
        arquivo.write('\n')


def carregar_dataset(pasta_dados: str):
    """
    Lê todos os arquivos .json da pasta e monta o dataset.

    Retorna:
        X: np.ndarray (N × 73) com as features
        y: lista de rótulos (nomes dos gestos)
    """
    X, y = [], []

    arquivos = [f for f in os.listdir(pasta_dados) if f.endswith('.json')]
    if not arquivos:
        print(f"[ERRO] Nenhum arquivo .json encontrado em: {pasta_dados}")
        print("   Execute primeiro: python training/coletar_dados.py --gesto A")
        sys.exit(1)

    print("[INFO] Carregando dados:")
    for arquivo in sorted(arquivos):
        nome_gesto = arquivo.replace('.json', '')
        caminho = os.path.join(pasta_dados, arquivo)

        with open(caminho, 'r', encoding='utf-8') as f:
            amostras = json.load(f)

        X.extend(amostras)
        y.extend([nome_gesto] * len(amostras))
        print(f"   {nome_gesto}: {len(amostras)} amostras")

    try:
        matriz = np.asarray(X, dtype=np.float32)
    except (TypeError, ValueError) as erro:
        raise ValueError(f'Dataset possui amostras com formato inválido: {erro}') from erro
    if matriz.ndim != 2 or matriz.shape[1] != 73:
        dimensao = matriz.shape[1] if matriz.ndim == 2 else 'irregular'
        raise ValueError(f'Dataset deve ter 73 features por amostra (encontrado: {dimensao}).')
    if not np.isfinite(matriz).all():
        raise ValueError('Dataset possui valores não finitos (NaN ou infinito).')

    print(f"\n   Total: {len(X)} amostras, {len(set(y))} gestos\n")
    return matriz, y


def avaliar_modelo(classificador, X, y, pasta_dados, manifesto):
    """
    Avalia o modelo com validação cruzada e exibe métricas detalhadas.
    Isso é essencial para o TCC: mostra que o sistema foi avaliado corretamente.
    """
    print("=" * 55)
    print("[INFO] AVALIAÇÃO DO MODELO")
    print("=" * 55)

    try:
        x_grupo, y_grupo, grupos = carregar_dataset_por_participante(pasta_dados, manifesto)
        resultado = avaliar_svm_por_participante(x_grupo, y_grupo, grupos)
        print("\nAvaliação por participante:")
        print(f"   Participantes: {', '.join(resultado['participantes'])}")
        print(f"   Acurácia: {resultado['acuracia'] * 100:.1f}%")
        print(f"   F1 macro: {resultado['f1_macro'] * 100:.1f}%")
        for gesto, metricas in resultado['por_gesto'].items():
            print(f"   {gesto}: precisão {metricas['precisao'] * 100:.1f}% | "
                  f"recall {metricas['recall'] * 100:.1f}% | F1 {metricas['f1'] * 100:.1f}%")
        return {
            'acuracia': resultado['acuracia'],
            'f1_macro': resultado['f1_macro'],
            'criterio_metrica': 'por participante',
            'por_gesto': resultado['por_gesto'],
            'matriz_confusao': resultado['matriz_confusao'],
        }
    except ValueError as erro:
        motivos_esperados = (
            'Nenhum manifesto de coleta foi encontrado.',
            'Colete todos os gestos com pelo menos 2 participantes para avaliar por pessoa.',
            'O manifesto não possui lotes de amostras.',
        )
        if str(erro) not in motivos_esperados:
            raise
        print('\nAviso: ainda não há cobertura suficiente para avaliar por participante.')
        print('A métrica abaixo é por amostra e não mede pessoas novas.')

    # Divide em treino (80%) e teste (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Treina com os dados de treino
    classificador.treinar(X_train, y_train)

    # Prediz no conjunto de teste
    from sklearn.preprocessing import LabelEncoder
    le = classificador.label_encoder
    y_pred_enc = classificador.modelo.predict(X_test.reshape(len(X_test), -1))

    # Converte de volta para nomes
    y_pred = le.inverse_transform(y_pred_enc)
    y_test_nomes = np.array(y_test)
    rotulos = sorted(set(y_test_nomes))
    matriz = confusion_matrix(y_test_nomes, y_pred, labels=rotulos).tolist()

    # Relatório de classificação (precisão, recall, F1 por classe)
    print("\n[INFO] Relatório por gesto:")
    relatorio = classification_report(y_test_nomes, y_pred, output_dict=True, zero_division=0)
    print(classification_report(y_test_nomes, y_pred, zero_division=0))

    # Acurácia geral
    acuracia = np.mean(y_pred == y_test_nomes)
    print(f"[OK] Acurácia no conjunto de teste: {acuracia * 100:.1f}%")

    # Validação cruzada (5-fold): mais confiável que uma única divisão
    # (requer retreinar com X completo para CV)
    print("\n[INFO] Validação cruzada (5-fold) com todos os dados:")
    from sklearn.base import clone
    modelo_cv = clone(classificador.modelo)
    from sklearn.preprocessing import LabelEncoder
    le_cv = LabelEncoder()
    y_enc = le_cv.fit_transform(y)
    scores = cross_val_score(modelo_cv, X, y_enc, cv=5, scoring='accuracy')
    print(f"   Acurácia por fold: {[f'{s*100:.1f}%' for s in scores]}")
    print(f"   Média: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    return {
        'acuracia': float(acuracia),
        'f1_macro': float(f1_score(y_test_nomes, y_pred, average='macro', zero_division=0)),
        'criterio_metrica': 'por amostra',
        'por_gesto': {
            gesto: {
                'precisao': float(relatorio[gesto]['precision']),
                'recall': float(relatorio[gesto]['recall']),
                'f1': float(relatorio[gesto]['f1-score']),
                'amostras': int(relatorio[gesto]['support']),
            }
            for gesto in rotulos
        },
        'matriz_confusao': {'rotulos': rotulos, 'valores': matriz},
    }


def main():
    parser = argparse.ArgumentParser(description='Treina o modelo estático do SignGlove.')
    parser.add_argument('--saida', help='arquivo JSON para salvar o resumo das métricas')
    args = parser.parse_args()
    pasta_dados = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures')
    manifesto = os.path.join(os.path.dirname(__file__), '..', 'data', 'metadata', 'coletas.jsonl')
    pasta_modelos = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(pasta_modelos, exist_ok=True)

    # Carrega os dados coletados
    X, y = carregar_dataset(pasta_dados)

    # Treina e avalia o modelo
    print("[INFO] Iniciando treinamento com SVM (mais preciso)...\n")
    classificador = ClassificadorGestos(
        algoritmo='svm',
        confianca_minima=0.75,
        buffer_frames=8
    )

    metricas = avaliar_modelo(classificador, X, y, pasta_dados, manifesto)

    # Re-treina com TODOS os dados para salvar o modelo final
    print("\n[INFO] Retreinando com 100% dos dados para o modelo final...")
    classificador.treinar(X, y)

    # Salva o modelo
    caminho_modelo = os.path.join(pasta_modelos, 'modelo_libras.pkl')
    classificador.salvar(caminho_modelo)

    print(f"\n[OK] Pronto! Acurácia {metricas['criterio_metrica']} estimada: "
          f"{metricas['acuracia']*100:.1f}%")
    print(f"[OK] F1 macro: {metricas['f1_macro']*100:.1f}%")
    if args.saida:
        salvar_resumo_treino(
            args.saida, metricas['acuracia'], metricas['f1_macro'],
            metricas['criterio_metrica'], y, len(X), metricas['por_gesto'],
            metricas['matriz_confusao'])
        print(f"[OK] Resumo salvo em: {os.path.abspath(args.saida)}")
    print(f"   Para usar: python main.py")


if __name__ == '__main__':
    main()
