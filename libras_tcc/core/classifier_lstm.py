"""
core/classifier_lstm.py
-----------------------
Classificador LSTM para sinais DINÂMICOS de Libras (com movimento).

Por que LSTM?
  - Sinais como OBRIGADO, OI, COMO VAI envolvem movimento ao longo do tempo.
  - O LSTM (Long Short-Term Memory) é uma rede neural que "lembra" de frames
    anteriores — perfeito para sequências temporais.

Como funciona:
  - Grava N frames de landmarks enquanto o sinal é feito
  - A sequência (N × 73 features) é passada para o LSTM
  - O LSTM retorna a classe com maior probabilidade

Requisito: pip install tensorflow  (ou tensorflow-cpu se não tiver GPU)
"""

import numpy as np
import os
import pickle

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_DISPONIVEL = True
except ImportError:
    TF_DISPONIVEL = False


SEQUENCIA_FRAMES = 30   # quantos frames por sinal (30 ≈ 1 segundo a 30fps)
N_FEATURES = 73         # tamanho do vetor de features por frame


class ClassificadorLSTM:
    """
    Treina e usa LSTM para reconhecer sinais dinâmicos de Libras.

    Parâmetros:
        n_frames: número de frames por sequência
        confianca_minima: limiar para aceitar uma predição
    """

    def __init__(self, n_frames=SEQUENCIA_FRAMES, confianca_minima=0.80):
        if not TF_DISPONIVEL:
            raise ImportError(
                "TensorFlow não encontrado.\n"
                "Instale com: pip install tensorflow"
            )
        self.n_frames = n_frames
        self.confianca_minima = confianca_minima
        self.modelo = None
        self.classes = []
        self.treinado = False

        # Buffer circular: acumula os últimos N frames em tempo real
        self._buffer_frames = []

    def _construir_modelo(self, n_classes: int) -> keras.Model:
        """
        Arquitetura da rede LSTM.

        Entrada: (batch, n_frames, n_features) = (batch, 30, 73)
        Saída:   (batch, n_classes)

        Camadas:
          LSTM(128) → aprende padrões temporais nos frames
          Dropout   → evita overfitting (memorizar em vez de aprender)
          LSTM(64)  → refina os padrões
          Dense     → classifica nas N classes
        """
        modelo = keras.Sequential([
            # Primeira camada LSTM — return_sequences=True para empilhar outra
            layers.LSTM(128, return_sequences=True,
                        input_shape=(self.n_frames, N_FEATURES)),
            layers.Dropout(0.3),

            # Segunda camada LSTM
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.3),

            # Camada densa intermediária
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),

            # Saída: probabilidade de cada classe
            layers.Dense(n_classes, activation='softmax'),
        ])

        modelo.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return modelo

    def treinar(self, X: np.ndarray, y: np.ndarray, epochs=50):
        """
        Treina o modelo LSTM.

        X: array (N_sequencias, n_frames, n_features)
        y: array de inteiros (índice da classe)
        epochs: quantas vezes passar pelo dataset completo
        """
        self.classes = sorted(list(set(y.tolist() if hasattr(y, 'tolist') else y)))

        # Converte rótulos string → inteiros
        y_int = np.array([self.classes.index(c) for c in y])
        n_classes = len(self.classes)

        self.modelo = self._construir_modelo(n_classes)

        print(f"🧠 Arquitetura LSTM:")
        self.modelo.summary()

        # Callbacks para parar cedo se não melhorar
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,           # para se não melhorar por 10 epochs
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,            # reduz learning rate pela metade
                patience=5
            )
        ]

        # Divide 80% treino / 20% validação
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_int, test_size=0.2, random_state=42, stratify=y_int
        )

        print(f"\n🔁 Treinando LSTM ({n_classes} classes, {len(X_train)} sequências)...")
        historico = self.modelo.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )

        # Avalia no conjunto de validação
        _, acc_val = self.modelo.evaluate(X_val, y_val, verbose=0)
        print(f"\n✅ Acurácia na validação: {acc_val*100:.1f}%")
        self.treinado = True
        return historico

    def adicionar_frame(self, features: np.ndarray):
        """
        Adiciona um frame ao buffer circular em tempo real.
        Chame isso a cada frame da webcam.
        """
        if features is not None:
            self._buffer_frames.append(features)
            # Mantém apenas os últimos N frames
            if len(self._buffer_frames) > self.n_frames:
                self._buffer_frames.pop(0)

    def prever_buffer(self) -> tuple:
        """
        Tenta reconhecer um sinal com o buffer atual.

        Retorna (gesto, confiança) ou (None, 0) se buffer incompleto
        ou confiança baixa.
        """
        if not self.treinado or len(self._buffer_frames) < self.n_frames:
            return None, 0.0

        sequencia = np.array(self._buffer_frames[-self.n_frames:])
        sequencia = sequencia.reshape(1, self.n_frames, N_FEATURES)

        probs = self.modelo.predict(sequencia, verbose=0)[0]
        idx = np.argmax(probs)
        confianca = float(probs[idx])

        if confianca < self.confianca_minima:
            return None, confianca

        return self.classes[idx], confianca

    def limpar_buffer(self):
        """Limpa o buffer de frames (use ao iniciar um novo sinal)."""
        self._buffer_frames = []

    def salvar(self, pasta: str):
        """Salva modelo Keras + metadados."""
        os.makedirs(pasta, exist_ok=True)
        self.modelo.save(os.path.join(pasta, 'lstm_model.keras'))
        with open(os.path.join(pasta, 'lstm_meta.pkl'), 'wb') as f:
            pickle.dump({
                'classes': self.classes,
                'n_frames': self.n_frames,
                'confianca_minima': self.confianca_minima
            }, f)
        print(f"💾 Modelo LSTM salvo em: {pasta}/")

    def carregar(self, pasta: str):
        """Carrega modelo Keras + metadados."""
        model_path = os.path.join(pasta, 'lstm_model.keras')
        meta_path  = os.path.join(pasta, 'lstm_meta.pkl')

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        self.modelo = keras.models.load_model(model_path)
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        self.classes = meta['classes']
        self.n_frames = meta['n_frames']
        self.confianca_minima = meta['confianca_minima']
        self.treinado = True
        print(f"📂 LSTM carregado: {self.classes}")
