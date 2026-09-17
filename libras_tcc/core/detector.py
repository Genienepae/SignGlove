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
        max_hands: número máximo de mãos detectadas ao mesmo tempo.
        min_detection_confidence: limiar mínimo de confiança para detectar a mão (0 a 1)
        min_tracking_confidence: limiar mínimo de confiança para rastrear entre frames
    """

    def __init__(self, max_hands=2, min_detection_confidence=0.6,
                 min_tracking_confidence=0.6, model_complexity=1):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Inicializa o modelo MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,          # False = modo vídeo (mais rápido)
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._ultimo_pulso = None
        self.ultimas_maos = []

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
            maos = list(resultados.multi_hand_landmarks)
            # Com duas mãos na imagem, manter a mesma mão principal entre
            # frames evita que a classificação de sinais de uma mão oscile.
            mao = self._selecionar_mao_principal(maos)
            detectou = True

            # Sempre desenha todas as mãos encontradas. A classificação atual
            # continua usando a mão principal, pois o modelo salvo tem 73
            # características (uma mão); sinais que exigem duas mãos precisam
            # de coleta e treinamento próprios.
            self.ultimas_maos = [self._extrair_landmarks(item) for item in maos]
            self.desenhar_maos(frame_anotado, self.ultimas_maos)

            # Extrai os 21 pontos como lista de (x, y, z)
            landmarks_normalizados = self._extrair_landmarks(mao)
        else:
            self.ultimas_maos = []
            self._ultimo_pulso = None

        return landmarks_normalizados, frame_anotado, detectou

    def _selecionar_mao_principal(self, maos):
        """Escolhe a mão mais próxima da usada no frame anterior."""
        pulsos = np.array([
            [mao.landmark[0].x, mao.landmark[0].y] for mao in maos],
            dtype=np.float32,
        )
        if self._ultimo_pulso is None:
            indice = 0
        else:
            indice = int(np.argmin(np.linalg.norm(pulsos - self._ultimo_pulso, axis=1)))
        self._ultimo_pulso = pulsos[indice]
        return maos[indice]

    def desenhar_maos(self, frame, maos):
        """Desenha landmarks normalizados sem executar outra inferência."""
        altura, largura = frame.shape[:2]
        for landmarks in maos:
            pontos = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
            coordenadas = [
                (int(np.clip(x, 0, 1) * (largura - 1)),
                 int(np.clip(y, 0, 1) * (altura - 1)))
                for x, y, _ in pontos
            ]
            for inicio, fim in self.mp_hands.HAND_CONNECTIONS:
                cv2.line(frame, coordenadas[inicio], coordenadas[fim], (0, 230, 118), 2)
            for ponto in coordenadas:
                cv2.circle(frame, ponto, 3, (255, 255, 255), -1)
                cv2.circle(frame, ponto, 4, (0, 230, 118), 1)

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
