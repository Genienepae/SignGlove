"""
core/classifier.py
------------------
Classifica gestos com base nos features extraídos.

Usa dois métodos:
  1. KNN (K-Nearest Neighbors) — simples, funciona bem com poucos dados
  2. SVM (Support Vector Machine) — mais preciso com mais dados

Técnicas para evitar falsos positivos:
  - Limiar de confiança mínima: só aceita predição se a confiança for alta
  - Buffer temporal: exige que o mesmo gesto apareça por N frames seguidos
  - Distância mínima ao vizinho mais próximo (para KNN)
"""

import numpy as np
import pickle
import os
from collections import deque, Counter

# Sklearn é instalado com: pip install scikit-learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class ClassificadorGestos:
    """
    Treina e usa um modelo de reconhecimento de gestos.

    Parâmetros:
        algoritmo: 'knn' ou 'svm'
        confianca_minima: só reconhece se a confiança for >= esse valor (0 a 1)
        buffer_frames: quantos frames consecutivos o gesto deve aparecer para confirmar
    """

    def __init__(self, algoritmo='knn', confianca_minima=0.75, buffer_frames=8):
        self.algoritmo = algoritmo
        self.confianca_minima = confianca_minima
        self.buffer_frames = buffer_frames

        # Buffer para suavizar predições ao longo do tempo
        self._buffer = deque(maxlen=buffer_frames)
        self._ultima_predicao = None

        self.modelo = None
        self.label_encoder = LabelEncoder()
        self.treinado = False

    def treinar(self, X: np.ndarray, y: list):
        """
        Treina o modelo com os dados coletados.

        X: matriz (N_amostras × N_features)
        y: lista de rótulos (ex: ['A', 'A', 'B', 'B', ...])
        """
        if len(set(y)) < 2:
            raise ValueError("É necessário pelo menos 2 classes para treinar.")

        y_encoded = self.label_encoder.fit_transform(y)

        if self.algoritmo == 'knn':
            # KNN: classifica pelo voto dos 5 vizinhos mais próximos
            # StandardScaler é importante: normaliza features para mesma escala
            self.modelo = Pipeline([
                ('scaler', StandardScaler()),
                ('knn', KNeighborsClassifier(
                    n_neighbors=5,
                    weights='distance',    # vizinhos mais próximos têm mais peso
                    metric='euclidean'
                ))
            ])
        elif self.algoritmo == 'svm':
            # SVM: melhor para dados com muitas features
            self.modelo = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', SVC(
                    kernel='rbf',
                    probability=True,      # necessário para obter confiança
                    C=10,
                    gamma='scale'
                ))
            ])
        else:
            raise ValueError(f"Algoritmo '{self.algoritmo}' não suportado. Use 'knn' ou 'svm'.")

        self.modelo.fit(X, y_encoded)
        self.treinado = True
        print(f"✅ Modelo treinado com {len(X)} amostras e {len(set(y))} classes.")
        print(f"   Classes: {list(self.label_encoder.classes_)}")

    def prever(self, features: np.ndarray) -> tuple:
        """
        Faz uma predição para um único frame.

        Retorna:
            gesto: nome do gesto reconhecido (str) ou None se não tiver confiança
            confianca: float entre 0 e 1
            confirmado: True se o buffer de frames confirmou o gesto
        """
        if not self.treinado or features is None:
            return None, 0.0, False

        features_2d = features.reshape(1, -1)

        # Obtém probabilidades de cada classe
        probs = self.modelo.predict_proba(features_2d)[0]
        idx_melhor = np.argmax(probs)
        confianca = probs[idx_melhor]

        if confianca < self.confianca_minima:
            # Confiança baixa → adiciona "dúvida" ao buffer
            self._buffer.append(None)
            return None, float(confianca), False

        gesto_predito = self.label_encoder.inverse_transform([idx_melhor])[0]
        self._buffer.append(gesto_predito)

        # Verifica se a maioria do buffer concorda (suavização temporal)
        confirmado, gesto_confirmado = self._verificar_buffer()

        return gesto_predito, float(confianca), confirmado

    def _verificar_buffer(self) -> tuple:
        """
        Verifica se o buffer de frames tem um gesto dominante.
        Retorna (confirmado, gesto) onde confirmado é True se
        >70% dos frames no buffer concordam.
        """
        if len(self._buffer) < self.buffer_frames:
            return False, None

        # Conta apenas predições válidas (não None)
        validas = [g for g in self._buffer if g is not None]
        if not validas:
            return False, None

        mais_comum, contagem = Counter(validas).most_common(1)[0]
        porcentagem = contagem / self.buffer_frames

        if porcentagem >= 0.7:  # 70% dos frames concordam → confirma
            return True, mais_comum

        return False, None

    def salvar(self, caminho: str):
        """Salva o modelo treinado em disco."""
        dados = {
            'modelo': self.modelo,
            'label_encoder': self.label_encoder,
            'algoritmo': self.algoritmo,
            'confianca_minima': self.confianca_minima,
        }
        with open(caminho, 'wb') as f:
            pickle.dump(dados, f)
        print(f"💾 Modelo salvo em: {caminho}")

    def carregar(self, caminho: str):
        """Carrega um modelo previamente treinado."""
        if not os.path.exists(caminho):
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")

        with open(caminho, 'rb') as f:
            dados = pickle.load(f)

        self.modelo = dados['modelo']
        self.label_encoder = dados['label_encoder']
        self.algoritmo = dados['algoritmo']
        self.confianca_minima = dados['confianca_minima']
        self.treinado = True
        print(f"📂 Modelo carregado: {list(self.label_encoder.classes_)}")

    def resetar_buffer(self):
        """Limpa o buffer temporal (use ao trocar de gesto intencionalmente)."""
        self._buffer.clear()
        self._ultima_predicao = None
