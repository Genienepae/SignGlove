"""
core/detector.py
----------------
Módulo responsável por detectar a mão na webcam usando MediaPipe
e extrair os 21 pontos (landmarks) em formato normalizado.

MediaPipe numera os pontos assim:
  0 = WRIST (pulso)
  1-4 = Polegar (THUMB)
  5-8 = Indicador (INDEX)
  9-12 = Médio (MIDDLE)
  13-16 = Anelar (RING)
  17-20 = Mínimo (PINKY)
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Detecta a mão em um frame de vídeo e retorna os landmarks normalizados.
    
    Parâmetros:
        max_hands: número máximo de mãos detectadas ao mesmo tempo (1 é suficiente para Libras)
        min_detection_confidence: limiar mínimo de confiança para detectar a mão (0 a 1)
        min_tracking_confidence: limiar mínimo de confiança para rastrear entre frames
    """

    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Inicializa o modelo MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          # False = modo vídeo (mais rápido)
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame):
        """
        Processa um frame BGR (OpenCV) e retorna os landmarks.

        Retorna:
            landmarks_normalizados: lista de 21 pontos (x, y, z) normalizados [0..1]
            frame_anotado: frame com os pontos desenhados (para visualização)
            detectou: True se detectou mão, False caso contrário
        """
        # MediaPipe trabalha com RGB; OpenCV usa BGR → converter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # melhora desempenho
        resultados = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        frame_anotado = frame.copy()
        landmarks_normalizados = None
        detectou = False

        if resultados.multi_hand_landmarks:
            # Pega a primeira mão detectada
            mao = resultados.multi_hand_landmarks[0]
            detectou = True

            # Desenha os pontos e conexões no frame
            self.mp_draw.draw_landmarks(
                frame_anotado,
                mao,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_styles.get_default_hand_landmarks_style(),
                self.mp_styles.get_default_hand_connections_style(),
            )

            # Extrai os 21 pontos como lista de (x, y, z)
            landmarks_normalizados = self._extrair_landmarks(mao)

        return landmarks_normalizados, frame_anotado, detectou

    def _extrair_landmarks(self, hand_landmarks):
        """
        Converte os landmarks do MediaPipe para uma lista simples de 63 valores:
        [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]

        Os valores já são normalizados pelo MediaPipe entre 0 e 1
        em relação ao tamanho do frame.
        """
        pontos = []
        for landmark in hand_landmarks.landmark:
            pontos.extend([landmark.x, landmark.y, landmark.z])
        return np.array(pontos, dtype=np.float32)

    def release(self):
        """Libera recursos do MediaPipe."""
        self.hands.close()
