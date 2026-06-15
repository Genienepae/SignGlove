"""
training/coletar_dinamico.py
----------------------------
Coleta amostras de sinais DINÂMICOS (com movimento) para o LSTM.

Como funciona:
  1. Você vê uma contagem regressiva (3, 2, 1...)
  2. Faz o sinal completo — o programa grava 30 frames de landmarks
  3. Repete N vezes (uma sequência por repetição)

Como usar:
    python training/coletar_dinamico.py --gesto OBRIGADO --repeticoes 60
    python training/coletar_dinamico.py --gesto OI --repeticoes 60

Dica de TCC: 60+ repetições por sinal são suficientes para começar.
Com augmentation (3×), isso vira 180 amostras por sinal.
"""

import cv2
import numpy as np
import argparse
import os
import json
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.detector import HandDetector
from core.features import extrair_features

N_FRAMES = 30          # frames por sequência (≈1 segundo a 30fps)
CONTAGEM_REGRESSIVA = 3  # segundos antes de cada gravação


def coletar_sequencias(nome_gesto: str, n_repeticoes: int, pasta_saida: str):
    """
    Grava N_FRAMES frames de landmarks para cada repetição do sinal.
    """
    os.makedirs(pasta_saida, exist_ok=True)
    caminho = os.path.join(pasta_saida, f"{nome_gesto}.json")

    # Carrega existentes
    sequencias = []
    if os.path.exists(caminho):
        with open(caminho, 'r') as f:
            sequencias = json.load(f)
        print(f"📁 {len(sequencias)} sequências existentes para '{nome_gesto}'")

    detector = HandDetector(min_detection_confidence=0.8)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam não encontrada.")
        return

    novas = []
    rep_atual = 0

    print(f"\n🎯 Coletando sinal dinâmico: '{nome_gesto}'")
    print(f"   Meta: {n_repeticoes} repetições × {N_FRAMES} frames cada")
    print(f"\n   Pressione ESPAÇO para iniciar a coleta\n")

    estado = 'aguardando'   # aguardando → contagem → gravando → fim
    t_estado = time.time()
    frames_gravados = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        landmarks, frame_anotado, detectou = detector.detect(frame)
        h, w = frame_anotado.shape[:2]

        # Painel de fundo
        cv2.rectangle(frame_anotado, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.rectangle(frame_anotado, (0, h - 55), (w, h), (20, 20, 20), -1)

        # Info do gesto
        cv2.putText(frame_anotado, f"Sinal: {nome_gesto}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 220, 0), 2)
        cv2.putText(frame_anotado, f"Repetições: {len(novas)}/{n_repeticoes}",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

        agora = time.time()

        # ── Máquina de estados ──────────────────────────────────────────

        if estado == 'aguardando':
            cv2.putText(frame_anotado, "ESPACO para gravar proxima repeticao",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 1)

        elif estado == 'contagem':
            restante = CONTAGEM_REGRESSIVA - int(agora - t_estado)
            if restante <= 0:
                estado = 'gravando'
                t_estado = agora
                frames_gravados = []
            else:
                # Número grande no centro
                cv2.putText(frame_anotado, str(restante),
                            (w // 2 - 30, h // 2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 200, 255), 6)
                cv2.putText(frame_anotado, "PREPARE-SE!",
                            (w // 2 - 110, h // 2 + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

        elif estado == 'gravando':
            # Barra de progresso
            pct = len(frames_gravados) / N_FRAMES
            barra_w = int((w - 20) * pct)
            cv2.rectangle(frame_anotado, (10, h - 35), (10 + barra_w, h - 15),
                          (0, 255, 100), -1)
            cv2.rectangle(frame_anotado, (10, h - 35), (w - 10, h - 15),
                          (80, 80, 80), 2)

            cv2.putText(frame_anotado, f"● GRAVANDO  {len(frames_gravados)}/{N_FRAMES}",
                        (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100), 2)

            # Grava o frame atual
            if detectou and landmarks is not None:
                f = extrair_features(landmarks)
                if f is not None:
                    frames_gravados.append(f.tolist())
            else:
                # Sem mão: frame zerado (sinaliza ausência)
                frames_gravados.append([0.0] * 73)

            if len(frames_gravados) >= N_FRAMES:
                novas.append(frames_gravados)
                rep_atual = len(novas)
                estado = 'resultado'
                t_estado = agora
                print(f"  ✅ Repetição {rep_atual}/{n_repeticoes} gravada")

                if len(novas) >= n_repeticoes:
                    estado = 'concluido'

        elif estado == 'resultado':
            cv2.putText(frame_anotado, f"✓ Repetição {rep_atual} gravada!",
                        (w // 2 - 170, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
            cv2.putText(frame_anotado, "ESPACO para continuar",
                        (w // 2 - 140, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        elif estado == 'concluido':
            cv2.putText(frame_anotado, f"🎉 CONCLUIDO! {n_repeticoes} repetições",
                        (w // 2 - 230, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)
            cv2.putText(frame_anotado, "Pressione Q para salvar e sair",
                        (w // 2 - 175, h // 2 + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("Coleta Dinâmica - Libras TCC", frame_anotado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(' '):
            if estado in ('aguardando', 'resultado'):
                estado = 'contagem'
                t_estado = time.time()
        elif tecla in [ord('q'), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    if novas:
        todas = sequencias + novas
        with open(caminho, 'w') as f:
            json.dump(todas, f)
        print(f"\n💾 Salvo: {caminho}")
        print(f"   Total de sequências para '{nome_gesto}': {len(todas)}")
    else:
        print("\n⚠️ Nenhuma sequência nova gravada.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gesto', type=str, required=True,
                        help='Nome do sinal (ex: OBRIGADO, OI, COMO_VAI)')
    parser.add_argument('--repeticoes', type=int, default=60,
                        help='Número de repetições (padrão: 60)')
    parser.add_argument('--pasta', type=str, default='data/sequences')
    args = parser.parse_args()

    coletar_sequencias(args.gesto, args.repeticoes, args.pasta)
