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

# Cada cadeia começa no pulso (0) e termina na ponta do dedo.
CADEIAS_DEDOS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)

# Cores BGR do OpenCV, uma para cada dedo numerado de 1 a 10.
CORES_DEDOS = (
    (40, 40, 240),    # 1 vermelho
    (0, 150, 255),    # 2 laranja
    (0, 230, 230),    # 3 amarelo
    (0, 210, 70),     # 4 verde
    (235, 210, 0),    # 5 ciano
    (255, 120, 0),    # 6 azul
    (220, 40, 170),   # 7 roxo
    (180, 0, 255),    # 8 rosa
    (150, 220, 150),  # 9 verde-claro
    (40, 110, 165),   # 10 marrom
)


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
        self.ultimos_lados = []

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
            self.ultimos_lados = self._extrair_lados(resultados, len(maos))
            self.desenhar_maos(frame_anotado, self.ultimas_maos, self.ultimos_lados)

            # Extrai os 21 pontos como lista de (x, y, z)
            landmarks_normalizados = self._extrair_landmarks(mao)
        else:
            self.ultimas_maos = []
            self.ultimos_lados = []
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

    @staticmethod
    def _extrair_lados(resultados, quantidade):
        """Obtém Right/Left do MediaPipe; mantém desenho útil se faltar dado."""
        classificacoes = getattr(resultados, 'multi_handedness', None) or []
        lados = []
        for classificacao in classificacoes:
            classes = getattr(classificacao, 'classification', [])
            lado = classes[0].label if classes else 'Right'
            lados.append(lado)
        return (lados + ['Right'] * quantidade)[:quantidade]

    def desenhar_maos(self, frame, maos, lados=None):
        """Desenha as duas mãos com um número e uma cor para cada dedo."""
        altura, largura = frame.shape[:2]
        lados = lados or ['Right'] * len(maos)
        for landmarks, lado in zip(maos, lados):
            pontos = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
            coordenadas = [
                (int(np.clip(x, 0, 1) * (largura - 1)),
                 int(np.clip(y, 0, 1) * (altura - 1)))
                for x, y, _ in pontos
            ]
            inicio_numero = 1 if lado == 'Right' else 6
            for indice_dedo, cadeia in enumerate(CADEIAS_DEDOS):
                numero = inicio_numero + indice_dedo
                cor = CORES_DEDOS[numero - 1]
                anterior = 0
                for atual in cadeia:
                    cv2.line(frame, coordenadas[anterior], coordenadas[atual], cor, 3)
                    cv2.circle(frame, coordenadas[atual], 5, cor, -1)
                    cv2.circle(frame, coordenadas[atual], 6, (255, 255, 255), 1)
                    anterior = atual

                ponta_x, ponta_y = coordenadas[cadeia[-1]]
                posicao = (ponta_x + 7, max(ponta_y - 7, 18))
                cv2.putText(frame, str(numero), posicao,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                cv2.putText(frame, str(numero), posicao,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, cor, 1)

            pulso_x, pulso_y = coordenadas[0]
            texto_lado = 'DIREITA 1-5' if lado == 'Right' else 'ESQUERDA 6-10'
            cv2.putText(frame, texto_lado, (pulso_x + 8, pulso_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
            cv2.putText(frame, texto_lado, (pulso_x + 8, pulso_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)

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
