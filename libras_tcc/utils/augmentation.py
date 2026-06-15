"""
utils/augmentation.py
---------------------
Aumentação de dados: multiplica o dataset sem gravar mais amostras.

Por que aumentar dados?
  - Coletar 200 amostras é trabalhoso.
  - Técnicas de augmentation criam variações artificiais realistas,
    aumentando o dataset de 200 → 1000+ amostras.
  - Isso melhora MUITO a generalização do modelo.

Técnicas aplicadas:
  1. Espelhamento horizontal (mão direita ↔ esquerda)
  2. Rotação leve dos landmarks (±10°)
  3. Escala aleatória (±10%)
  4. Ruído gaussiano (simula imprecisão do sensor)
  5. Translação aleatória (mão em posições ligeiramente diferentes)
"""

import numpy as np


def aumentar_dataset(X: np.ndarray, y: list, fator: int = 4) -> tuple:
    """
    Recebe o dataset original e retorna uma versão aumentada.

    Parâmetros:
        X: (N, 73) — features originais
        y: lista de rótulos
        fator: quantas vezes aumentar cada amostra (padrão: 4×)

    Retorna:
        X_aug: (N * fator, 73)
        y_aug: lista de rótulos correspondente
    """
    X_aug = list(X)
    y_aug = list(y)

    for i, (features, rotulo) in enumerate(zip(X, y)):
        for _ in range(fator - 1):  # -1 porque o original já está incluído
            variacao = _aplicar_augmentation(features)
            X_aug.append(variacao)
            y_aug.append(rotulo)

    X_aug = np.array(X_aug, dtype=np.float32)
    print(f"📈 Dataset aumentado: {len(X)} → {len(X_aug)} amostras ({fator}×)")
    return X_aug, y_aug


def _aplicar_augmentation(features: np.ndarray) -> np.ndarray:
    """
    Aplica uma combinação aleatória de transformações.
    Cada chamada gera uma variação diferente.
    """
    # Reshape para (21, 3) para manipular os pontos individualmente
    pontos = features[:63].reshape(21, 3).copy()
    angulos = features[63:].copy()   # mantém os ângulos (últimas 10 features)

    # Sorteia quais transformações aplicar (sempre pelo menos 1)
    transformacoes = np.random.choice(
        ['ruido', 'rotacao', 'escala', 'translacao', 'espelho'],
        size=np.random.randint(1, 4),
        replace=False
    )

    for t in transformacoes:
        if t == 'ruido':
            pontos = _adicionar_ruido(pontos)
        elif t == 'rotacao':
            pontos = _rotacionar(pontos)
        elif t == 'escala':
            pontos = _escalar(pontos)
        elif t == 'translacao':
            pontos = _transladar(pontos)
        elif t == 'espelho':
            pontos = _espelhar_horizontal(pontos)

    # Adiciona ruído leve nos ângulos também
    angulos += np.random.normal(0, 1.5, size=angulos.shape)  # ±1.5 graus

    return np.concatenate([pontos.flatten(), angulos]).astype(np.float32)


def _adicionar_ruido(pontos: np.ndarray, intensidade=0.015) -> np.ndarray:
    """
    Adiciona ruído gaussiano — simula imprecisão do MediaPipe.
    intensidade=0.015 significa ±1.5% do tamanho da mão.
    """
    ruido = np.random.normal(0, intensidade, pontos.shape)
    return pontos + ruido


def _rotacionar(pontos: np.ndarray, angulo_max_graus=10) -> np.ndarray:
    """
    Rotaciona todos os pontos em torno do pulso (ponto 0).
    Simula a mão levemente inclinada.
    """
    angulo = np.radians(np.random.uniform(-angulo_max_graus, angulo_max_graus))
    cos_a, sin_a = np.cos(angulo), np.sin(angulo)

    # Rotação 2D no plano X-Y (mais relevante para câmera frontal)
    x = pontos[:, 0].copy()
    y = pontos[:, 1].copy()
    pontos[:, 0] = cos_a * x - sin_a * y
    pontos[:, 1] = sin_a * x + cos_a * y
    return pontos


def _escalar(pontos: np.ndarray, variacao=0.10) -> np.ndarray:
    """
    Escala a mão para cima ou para baixo.
    Simula a mão mais perto ou longe da câmera.
    """
    fator = np.random.uniform(1 - variacao, 1 + variacao)
    return pontos * fator


def _transladar(pontos: np.ndarray, variacao=0.05) -> np.ndarray:
    """
    Desloca todos os pontos levemente.
    Simula a mão em posições ligeiramente diferentes do frame.
    """
    delta = np.random.uniform(-variacao, variacao, size=(1, 3))
    return pontos + delta


def _espelhar_horizontal(pontos: np.ndarray) -> np.ndarray:
    """
    Espelha a mão no eixo X.
    Converte mão direita ↔ esquerda.
    Muito útil se você coletou dados só com uma mão.
    """
    pontos_espelho = pontos.copy()
    pontos_espelho[:, 0] = -pontos_espelho[:, 0]   # inverte o eixo X
    return pontos_espelho


def aumentar_sequencias(X_seq: np.ndarray, y: list, fator: int = 3) -> tuple:
    """
    Versão da aumentação para sequências (usado no LSTM).

    X_seq: (N, n_frames, n_features)
    """
    X_aug = list(X_seq)
    y_aug = list(y)

    for seq, rotulo in zip(X_seq, y):
        for _ in range(fator - 1):
            # Aplica augmentation frame a frame dentro da sequência
            seq_aug = np.array([
                _aplicar_augmentation(frame) for frame in seq
            ])
            X_aug.append(seq_aug)
            y_aug.append(rotulo)

    X_aug = np.array(X_aug, dtype=np.float32)
    print(f"📈 Sequências aumentadas: {len(X_seq)} → {len(X_aug)} ({fator}×)")
    return X_aug, y_aug
