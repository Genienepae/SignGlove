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
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.classifier import ClassificadorGestos

# Sklearn para métricas de avaliação
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


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
        print(f"❌ Nenhum arquivo .json encontrado em: {pasta_dados}")
        print("   Execute primeiro: python training/coletar_dados.py --gesto A")
        sys.exit(1)

    print("📂 Carregando dados:")
    for arquivo in sorted(arquivos):
        nome_gesto = arquivo.replace('.json', '')
        caminho = os.path.join(pasta_dados, arquivo)

        with open(caminho, 'r') as f:
            amostras = json.load(f)

        X.extend(amostras)
        y.extend([nome_gesto] * len(amostras))
        print(f"   {nome_gesto}: {len(amostras)} amostras")

    print(f"\n   Total: {len(X)} amostras, {len(set(y))} gestos\n")
    return np.array(X, dtype=np.float32), y


def avaliar_modelo(classificador, X, y):
    """
    Avalia o modelo com validação cruzada e exibe métricas detalhadas.
    Isso é essencial para o TCC: mostra que o sistema foi avaliado corretamente.
    """
    print("=" * 55)
    print("📊 AVALIAÇÃO DO MODELO")
    print("=" * 55)

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

    # Relatório de classificação (precisão, recall, F1 por classe)
    print("\n📋 Relatório por gesto:")
    print(classification_report(y_test_nomes, y_pred))

    # Acurácia geral
    acuracia = np.mean(y_pred == y_test_nomes)
    print(f"✅ Acurácia no conjunto de teste: {acuracia * 100:.1f}%")

    # Validação cruzada (5-fold): mais confiável que uma única divisão
    # (requer retreinar com X completo para CV)
    print("\n🔄 Validação cruzada (5-fold) com todos os dados:")
    from sklearn.base import clone
    modelo_cv = clone(classificador.modelo)
    from sklearn.preprocessing import LabelEncoder
    le_cv = LabelEncoder()
    y_enc = le_cv.fit_transform(y)
    scores = cross_val_score(modelo_cv, X, y_enc, cv=5, scoring='accuracy')
    print(f"   Acurácia por fold: {[f'{s*100:.1f}%' for s in scores]}")
    print(f"   Média: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")

    return acuracia


def main():
    pasta_dados = os.path.join(os.path.dirname(__file__), '..', 'data', 'gestures')
    pasta_modelos = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(pasta_modelos, exist_ok=True)

    # Carrega os dados coletados
    X, y = carregar_dataset(pasta_dados)

    # Treina e avalia o modelo
    print("🤖 Iniciando treinamento com SVM (mais preciso)...\n")
    classificador = ClassificadorGestos(
        algoritmo='svm',
        confianca_minima=0.75,
        buffer_frames=8
    )

    acuracia = avaliar_modelo(classificador, X, y)

    # Re-treina com TODOS os dados para salvar o modelo final
    print("\n🔁 Retreinando com 100% dos dados para o modelo final...")
    classificador.treinar(X, y)

    # Salva o modelo
    caminho_modelo = os.path.join(pasta_modelos, 'modelo_libras.pkl')
    classificador.salvar(caminho_modelo)

    print(f"\n🎉 Pronto! Acurácia final estimada: {acuracia*100:.1f}%")
    print(f"   Para usar: python main.py")


if __name__ == '__main__':
    main()
