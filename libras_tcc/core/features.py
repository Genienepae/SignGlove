"""
core/features.py
----------------
Extrai características robustas dos 21 landmarks da mão.

Por que normalizar os landmarks?
  - Se você mover a mão para a esquerda ou direita, os valores x,y mudam.
  - Se o modelo aprender com mão em posições fixas, ele vai errar quando você
    mover a mão. A solução é extrair características RELATIVAS à própria mão.

Estratégia usada:
  1. Centralizar: subtrair a posição do pulso (ponto 0) de todos os pontos
  2. Escalar: dividir pela maior distância encontrada (torna independente do tamanho)
  3. Ângulos entre dedos: features extras que melhoram muito a precisão
"""

import numpy as np


# Índices dos landmarks no MediaPipe (para facilitar a leitura do código)
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20
INDEX_MCP = 5   # base do indicador


def extrair_features(landmarks_raw: np.ndarray) -> np.ndarray:
    """
    Recebe os 63 valores brutos (21 pontos × 3 coordenadas)
    e retorna um vetor de features normalizado e invariante à posição/escala.

    Retorna: np.ndarray de tamanho ~73 (63 + 10 ângulos)
    """
    if landmarks_raw is None:
        return None

    # Reshape: (63,) → (21, 3)
    pontos = landmarks_raw.reshape(21, 3)

    # --- Passo 1: centralizar em relação ao pulso ---
    pulso = pontos[WRIST].copy()
    pontos_centrados = pontos - pulso          # agora o pulso fica em (0,0,0)

    # --- Passo 2: normalizar pela escala da mão ---
    # Usamos a distância do pulso até a base do indicador como referência
    distancia_ref = np.linalg.norm(pontos_centrados[INDEX_MCP])
    if distancia_ref < 1e-6:                   # evitar divisão por zero
        distancia_ref = 1.0
    pontos_norm = pontos_centrados / distancia_ref

    # --- Passo 3: calcular ângulos de flexão de cada dedo ---
    angulos = calcular_angulos_dedos(pontos_norm)

    # Achata (21×3 = 63) e concatena com os 10 ângulos → 73 features no total
    features = np.concatenate([pontos_norm.flatten(), angulos])
    return features.astype(np.float32)


def calcular_angulos_dedos(pontos: np.ndarray) -> np.ndarray:
    """
    Calcula o ângulo de flexão das juntas dos 5 dedos.
    Cada dedo tem 2 ângulos (MCP→PIP e PIP→DIP), totalizando 10 valores.

    Isso é muito útil para diferenciar sinais como A, E, S em Libras
    que dependem de quanto os dedos estão dobrados.
    """
    # Defina os triplos de pontos para cada junta: (base, meio, ponta)
    juntas = [
        # Polegar
        (1, 2, 3), (2, 3, 4),
        # Indicador
        (5, 6, 7), (6, 7, 8),
        # Médio
        (9, 10, 11), (10, 11, 12),
        # Anelar
        (13, 14, 15), (14, 15, 16),
        # Mínimo
        (17, 18, 19), (18, 19, 20),
    ]

    angulos = []
    for (a, b, c) in juntas:
        angulo = _angulo_entre_tres_pontos(pontos[a], pontos[b], pontos[c])
        angulos.append(angulo)

    return np.array(angulos, dtype=np.float32)


def _angulo_entre_tres_pontos(p1, p2, p3) -> float:
    """
    Calcula o ângulo em graus formado por p1 → p2 → p3.
    Usa produto escalar (dot product): cos(θ) = (v1·v2) / (|v1||v2|)
    """
    v1 = p1 - p2   # vetor do meio para trás
    v2 = p3 - p2   # vetor do meio para frente

    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 < 1e-6 or norm_v2 < 1e-6:
        return 0.0

    cos_theta = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)    # evitar erros numéricos
    angulo_rad = np.arccos(cos_theta)
    return float(np.degrees(angulo_rad))


def dedos_levantados(landmarks_raw: np.ndarray) -> list:
    """
    Retorna uma lista [polegar, indicador, médio, anelar, mínimo]
    onde True = dedo levantado, False = dedo dobrado.

    Útil para regras simples e para depuração visual.
    """
    if landmarks_raw is None:
        return [False] * 5

    pontos = landmarks_raw.reshape(21, 3)

    levantados = []

    # Polegar: compara x (horizontal) com o ponto anterior
    # (lógica diferente porque o polegar dobra lateralmente)
    levantados.append(pontos[THUMB_TIP][0] > pontos[3][0])

    # Demais dedos: ponta mais alta (y menor) que a segunda falange
    for ponta, segunda_falange in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        levantados.append(pontos[ponta][1] < pontos[segunda_falange][1])

    return levantados
