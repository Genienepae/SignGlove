"""
training/coletar_dados.py
--------------------------
Modo de treinamento: abre a webcam e coleta amostras de um gesto específico.

Como usar:
    python training/coletar_dados.py --gesto A --amostras 200

Cada letra/sinal de Libras precisa de pelo menos 100-200 amostras para
o modelo aprender com boa precisão.

Dica para um TCC sólido:
  - Colete em diferentes fundos (parede branca, azul, etc.)
  - Colete com diferentes intensidades de luz
  - Peça para 2-3 pessoas diferentes coletarem amostras
  - Isso se chama "diversidade de dados" e melhora muito a precisão
"""

import cv2
import numpy as np
import argparse
import os
import json
import sys

# Adiciona a raiz do projeto ao path para importar os módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.detector import HandDetector
from core.features import extrair_features, dedos_levantados


def coletar_amostras(nome_gesto: str, qtd_amostras: int, pasta_saida: str):
    """
    Abre a webcam e coleta amostras do gesto especificado.

    Os dados são salvos em:
      data/gestures/<nome_gesto>.json
    """
    os.makedirs(pasta_saida, exist_ok=True)
    caminho_saida = os.path.join(pasta_saida, f"{nome_gesto}.json")

    # Carrega amostras já existentes (para poder adicionar mais depois)
    amostras_existentes = []
    if os.path.exists(caminho_saida):
        with open(caminho_saida, 'r') as f:
            amostras_existentes = json.load(f)
        print(f"📁 Encontradas {len(amostras_existentes)} amostras anteriores para '{nome_gesto}'")

    detector = HandDetector(min_detection_confidence=0.8)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Erro: não foi possível abrir a webcam.")
        return

    novas_amostras = []
    coletando = False   # começa pausado; usuário pressiona ESPAÇO para iniciar

    print(f"\n🎯 Coletando amostras para o gesto: '{nome_gesto}'")
    print(f"   Meta: {qtd_amostras} amostras novas")
    print(f"\nControles:")
    print(f"  ESPAÇO  → iniciar/pausar coleta")
    print(f"  Q       → encerrar e salvar\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # espelha para parecer um espelho
        landmarks, frame_anotado, detectou = detector.detect(frame)

        # --- Interface visual ---
        h, w = frame_anotado.shape[:2]

        # Fundo do status bar
        cv2.rectangle(frame_anotado, (0, 0), (w, 80), (30, 30, 30), -1)

        # Gesto atual
        cv2.putText(frame_anotado, f"Gesto: {nome_gesto}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Progresso
        total = len(amostras_existentes) + len(novas_amostras)
        progresso = f"Amostras: {len(novas_amostras)}/{qtd_amostras} novas  |  Total: {total}"
        cv2.putText(frame_anotado, progresso,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Status de coleta
        if coletando:
            status_cor = (0, 255, 0)
            status_txt = "● COLETANDO"
        else:
            status_cor = (0, 165, 255)
            status_txt = "❚❚ PAUSADO — pressione ESPAÇO"

        cv2.putText(frame_anotado, status_txt,
                    (w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_cor, 2)

        # Mostra quais dedos estão levantados (ajuda a conferir o gesto)
        if detectou and landmarks is not None:
            dedos = dedos_levantados(landmarks)
            nomes = ['P', 'I', 'M', 'A', 'Mi']  # Polegar, Indicador, Médio, Anelar, Mínimo
            for i, (nome, levantado) in enumerate(zip(nomes, dedos)):
                cor = (0, 255, 100) if levantado else (60, 60, 60)
                cv2.circle(frame_anotado, (10 + i * 35, h - 20), 12, cor, -1)
                cv2.putText(frame_anotado, nome, (4 + i * 35, h - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

            # Coleta a amostra se estiver no modo ativo
            if coletando:
                features = extrair_features(landmarks)
                if features is not None:
                    novas_amostras.append(features.tolist())

                    # Barra de progresso visual
                    pct = len(novas_amostras) / qtd_amostras
                    barra_w = int(w * pct)
                    cv2.rectangle(frame_anotado, (0, h - 5), (barra_w, h), (0, 255, 0), -1)

                    if len(novas_amostras) >= qtd_amostras:
                        print(f"\n✅ Meta atingida! {qtd_amostras} amostras coletadas.")
                        coletando = False
        else:
            cv2.putText(frame_anotado, "Mão não detectada",
                        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Coleta de Dados - Libras TCC", frame_anotado)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord(' '):
            coletando = not coletando
            estado = "INICIADA" if coletando else "PAUSADA"
            print(f"Coleta {estado}. Amostras novas até agora: {len(novas_amostras)}")
        elif tecla == ord('q') or tecla == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.release()

    # Salva as amostras
    if novas_amostras:
        todas = amostras_existentes + novas_amostras
        with open(caminho_saida, 'w') as f:
            json.dump(todas, f)
        print(f"\n💾 Salvo: {caminho_saida}")
        print(f"   Total de amostras para '{nome_gesto}': {len(todas)}")
    else:
        print("\n⚠️  Nenhuma amostra nova coletada.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coleta amostras de gestos de Libras')
    parser.add_argument('--gesto', type=str, required=True,
                        help='Nome do gesto (ex: A, B, OI, OBRIGADO)')
    parser.add_argument('--amostras', type=int, default=200,
                        help='Quantidade de amostras a coletar (padrão: 200)')
    parser.add_argument('--pasta', type=str, default='data/gestures',
                        help='Pasta onde salvar os dados')
    args = parser.parse_args()

    coletar_amostras(args.gesto, args.amostras, args.pasta)
