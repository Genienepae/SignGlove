"""
training/treinar_lstm.py
------------------------
Treina o modelo LSTM para sinais dinâmicos de Libras.

Fluxo:
  1. Carrega as sequências de data/sequences/
  2. Aplica aumentação de dados (3×)
  3. Treina o LSTM com early stopping
  4. Gera gráficos de acurácia/loss
  5. Salva o modelo em models/lstm/

Como usar:
    python training/treinar_lstm.py
"""

import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.classifier_lstm import ClassificadorLSTM, N_FEATURES, SEQUENCIA_FRAMES
from utils.augmentation import aumentar_sequencias


def carregar_sequencias(pasta: str):
    """Carrega todos os arquivos .json de sequências."""
    X, y = [], []
    arquivos = [f for f in os.listdir(pasta) if f.endswith('.json')]

    if not arquivos:
        print(f"❌ Nenhuma sequência em: {pasta}")
        print("   Execute: python training/coletar_dinamico.py --gesto OBRIGADO")
        sys.exit(1)

    print("📂 Carregando sequências:")
    for arq in sorted(arquivos):
        nome = arq.replace('.json', '')
        with open(os.path.join(pasta, arq)) as f:
            seqs = json.load(f)

        # Garante que todas as sequências têm o tamanho certo
        for seq in seqs:
            if len(seq) == SEQUENCIA_FRAMES:
                X.append(seq)
                y.append(nome)
            else:
                # Padding ou corte se necessário
                if len(seq) < SEQUENCIA_FRAMES:
                    # Preenche com zeros no final
                    seq += [[0.0] * N_FEATURES] * (SEQUENCIA_FRAMES - len(seq))
                else:
                    seq = seq[:SEQUENCIA_FRAMES]
                X.append(seq)
                y.append(nome)

        print(f"   {nome}: {len(seqs)} sequências")

    X = np.array(X, dtype=np.float32)
    print(f"\n   Total: {len(X)} sequências, {len(set(y))} sinais\n")
    return X, y


def plotar_historico(historico, pasta_saida):
    """Gera e salva gráficos de treino — útil para o TCC."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # sem janela, só salva o arquivo
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Acurácia
        ax1.plot(historico.history['accuracy'], label='Treino')
        ax1.plot(historico.history['val_accuracy'], label='Validação')
        ax1.set_title('Acurácia por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acurácia')
        ax1.legend()
        ax1.grid(True)

        # Loss
        ax2.plot(historico.history['loss'], label='Treino')
        ax2.plot(historico.history['val_loss'], label='Validação')
        ax2.set_title('Loss por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        caminho_fig = os.path.join(pasta_saida, 'historico_treinamento.png')
        plt.savefig(caminho_fig, dpi=150)
        plt.close()
        print(f"📊 Gráfico salvo em: {caminho_fig}")

    except ImportError:
        print("⚠️  matplotlib não instalado — gráfico não gerado.")
        print("   Instale com: pip install matplotlib")


def main():
    pasta_seq   = os.path.join(os.path.dirname(__file__), '..', 'data', 'sequences')
    pasta_model = os.path.join(os.path.dirname(__file__), '..', 'models', 'lstm')
    os.makedirs(pasta_model, exist_ok=True)

    # 1. Carrega dados
    X, y = carregar_sequencias(pasta_seq)

    # 2. Aumentação de dados (3×): 60 sequências → 180 por sinal
    X_aug, y_aug = aumentar_sequencias(X, y, fator=3)

    # 3. Treina LSTM
    clf = ClassificadorLSTM(
        n_frames=SEQUENCIA_FRAMES,
        confianca_minima=0.80
    )
    historico = clf.treinar(X_aug, np.array(y_aug), epochs=80)

    # 4. Salva
    clf.salvar(pasta_model)

    # 5. Gráficos
    plotar_historico(historico, pasta_model)

    print("\n🎉 LSTM treinado com sucesso!")
    print(f"   Execute: python main.py")


if __name__ == '__main__':
    main()
