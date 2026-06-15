"""
main.py
-------
Aplicação principal: reconhecimento de gestos de Libras em tempo real.

Como usar:
    python main.py

Controles:
    Q ou ESC → sair
    C        → limpar o texto reconhecido
    ESPAÇO   → adicionar espaço ao texto
"""

import cv2
import numpy as np
import os
import time
import sys

from core.detector import HandDetector
from core.features import extrair_features, dedos_levantados
from core.classifier import ClassificadorGestos


# ─── Configurações ───────────────────────────────────────────────────────────

CAMINHO_MODELO = 'models/modelo_libras.pkl'
CONFIANCA_MINIMA = 0.75       # só mostra predição com >= 75% de confiança
BUFFER_FRAMES = 8             # frames necessários para confirmar um gesto
COOLDOWN_SEGUNDOS = 1.5       # tempo mínimo entre duas confirmações

# Cores (BGR)
VERDE    = (0, 220, 100)
AMARELO  = (0, 220, 255)
VERMELHO = (0, 0, 220)
BRANCO   = (255, 255, 255)
CINZA    = (150, 150, 150)
FUNDO    = (20, 20, 20)


# ─── Funções auxiliares ───────────────────────────────────────────────────────

def desenhar_hud(frame, gesto, confianca, confirmado, texto_acumulado, fps):
    """Desenha a interface sobreposta no frame da webcam."""
    h, w = frame.shape[:2]

    # Painel superior
    cv2.rectangle(frame, (0, 0), (w, 90), FUNDO, -1)

    # FPS (canto superior direito)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CINZA, 1)

    # Gesto atual e confiança
    if gesto:
        cor_gesto = VERDE if confirmado else AMARELO
        cv2.putText(frame, f"Gesto: {gesto}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1, cor_gesto, 2)

        # Barra de confiança
        barra_max = 200
        barra_atual = int(barra_max * confianca)
        cv2.rectangle(frame, (10, 50), (10 + barra_max, 68), (60, 60, 60), -1)
        cor_barra = VERDE if confianca >= 0.8 else AMARELO
        cv2.rectangle(frame, (10, 50), (10 + barra_atual, 68), cor_barra, -1)
        cv2.putText(frame, f"{confianca*100:.0f}%", (220, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, BRANCO, 1)

        if confirmado:
            cv2.putText(frame, "✓ CONFIRMADO", (300, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, VERDE, 2)
    else:
        cv2.putText(frame, "Aguardando gesto...",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, CINZA, 1)

    # Texto acumulado (parte inferior)
    cv2.rectangle(frame, (0, h - 70), (w, h), FUNDO, -1)
    cv2.putText(frame, "Texto:", (10, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, CINZA, 1)
    # Mostra só os últimos 30 caracteres para não ultrapassar a tela
    texto_exibido = texto_acumulado[-30:] if len(texto_acumulado) > 30 else texto_acumulado
    cv2.putText(frame, texto_exibido, (75, h - 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, BRANCO, 2)
    cv2.putText(frame, "[Q] Sair  [C] Limpar  [ESPACO] Espaco",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CINZA, 1)

    return frame


def main():
    # Verifica se o modelo foi treinado
    if not os.path.exists(CAMINHO_MODELO):
        print(f"❌ Modelo não encontrado: {CAMINHO_MODELO}")
        print("   Execute primeiro:")
        print("     python training/coletar_dados.py --gesto A")
        print("     python training/treinar_modelo.py")
        sys.exit(1)

    print("🚀 Iniciando sistema de reconhecimento de Libras...")

    # Inicializa componentes
    detector = HandDetector(
        max_hands=1,
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7
    )
    classificador = ClassificadorGestos(
        confianca_minima=CONFIANCA_MINIMA,
        buffer_frames=BUFFER_FRAMES
    )
    classificador.carregar(CAMINHO_MODELO)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("❌ Erro: webcam não encontrada.")
        sys.exit(1)

    texto_acumulado = ""
    ultimo_confirmado = ""
    ultimo_tempo_confirmacao = 0
    fps_contador = 0
    fps_tempo = time.time()
    fps_atual = 0

    print("✅ Sistema pronto! Pressione Q para sair.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        # FPS
        fps_contador += 1
        agora = time.time()
        if agora - fps_tempo >= 1.0:
            fps_atual = fps_contador
            fps_contador = 0
            fps_tempo = agora

        # Detecta mão e extrai landmarks
        landmarks, frame_anotado, detectou = detector.detect(frame)

        gesto_atual = None
        confianca_atual = 0.0
        confirmado_agora = False

        if detectou and landmarks is not None:
            features = extrair_features(landmarks)
            gesto_atual, confianca_atual, confirmado_agora = classificador.prever(features)

            # Adiciona ao texto se confirmado e passou o cooldown
            if (confirmado_agora and
                    gesto_atual and
                    gesto_atual != ultimo_confirmado and
                    (agora - ultimo_tempo_confirmacao) >= COOLDOWN_SEGUNDOS):
                texto_acumulado += gesto_atual
                ultimo_confirmado = gesto_atual
                ultimo_tempo_confirmacao = agora
                print(f"✅ Reconhecido: {gesto_atual} ({confianca_atual*100:.0f}%)")
        else:
            # Sem mão → reseta para evitar acúmulo indevido no buffer
            classificador.resetar_buffer()
            ultimo_confirmado = ""

        # Desenha interface
        frame_final = desenhar_hud(
            frame_anotado, gesto_atual, confianca_atual,
            confirmado_agora, texto_acumulado, fps_atual
        )

        cv2.imshow("Libras TCC - Reconhecimento em Tempo Real", frame_final)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in [ord('q'), 27]:    # Q ou ESC
            break
        elif tecla == ord('c'):
            texto_acumulado = ""
            print("🗑️  Texto limpo.")
        elif tecla == ord(' '):
            texto_acumulado += " "

    # Finaliza
    cap.release()
    cv2.destroyAllWindows()
    detector.release()
    print(f"\n📝 Texto final reconhecido: '{texto_acumulado}'")
    print("👋 Encerrando sistema.")


if __name__ == '__main__':
    main()
