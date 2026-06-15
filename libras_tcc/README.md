# 🤟 Sistema de Reconhecimento de Libras com IA

**TCC — Reconhecimento de Língua de Sinais Brasileira usando MediaPipe e Machine Learning**

---

## 📁 Estrutura do Projeto

```
libras_tcc/
│
├── main.py                     ← Aplicação principal (reconhecimento em tempo real)
│
├── core/
│   ├── detector.py             ← Detecta a mão e os 21 landmarks (MediaPipe)
│   ├── features.py             ← Extrai características normalizadas dos landmarks
│   └── classifier.py          ← Classifica gestos (KNN ou SVM) com anti-falsos-positivos
│
├── training/
│   ├── coletar_dados.py        ← Modo de coleta de amostras pela webcam
│   └── treinar_modelo.py       ← Treina o modelo e exibe métricas para o TCC
│
├── data/
│   └── gestures/               ← Arquivos .json com as amostras coletadas
│       ├── A.json
│       ├── B.json
│       └── ...
│
├── models/
│   └── modelo_libras.pkl       ← Modelo treinado (gerado automaticamente)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar (passo a passo)

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Coletar amostras de cada gesto
```bash
# Coleta 200 amostras da letra A
python training/coletar_dados.py --gesto A --amostras 200

# Repita para cada gesto que quiser reconhecer
python training/coletar_dados.py --gesto B --amostras 200
python training/coletar_dados.py --gesto C --amostras 200
python training/coletar_dados.py --gesto OI --amostras 200
```
> **Dica:** Pressione ESPAÇO para iniciar/pausar a coleta dentro do programa.

### 3. Treinar o modelo
```bash
python training/treinar_modelo.py
```
Isso vai exibir a acurácia, precisão e recall por gesto — dados importantes para o TCC.

### 4. Executar o reconhecimento em tempo real
```bash
python main.py
```

---

## 🧠 Como Funciona — Explicado para o TCC

### Os 21 Landmarks do MediaPipe

O MediaPipe detecta 21 pontos (landmarks) na mão:

```
        8   12  16  20
        |   |   |   |
    4   7   11  15  19
    |   6   10  14  18
    3   5---9---13--17
    |   |
    2   |
     \  |
      1-0 (pulso)
```

| Índice | Região       |
|--------|--------------|
| 0      | Pulso (WRIST) |
| 1–4    | Polegar      |
| 5–8    | Indicador    |
| 9–12   | Médio        |
| 13–16  | Anelar       |
| 17–20  | Mínimo       |

### Pipeline de Reconhecimento

```
Webcam → Frame BGR
    ↓
HandDetector (MediaPipe)
    → 21 landmarks (x, y, z) normalizados 0–1
    ↓
Extração de Features (features.py)
    → Centralização + normalização de escala
    → 10 ângulos de flexão dos dedos
    → Vetor final: 73 valores
    ↓
Classificador (SVM/KNN)
    → Probabilidade por classe
    → Filtro de confiança mínima (75%)
    → Buffer temporal (8 frames)
    ↓
Gesto confirmado
```

### Por que Normalizar os Landmarks?

Sem normalização, mover a mão para a esquerda muda os valores x de todos os pontos — o modelo confunde gestos iguais feitos em posições diferentes.

A solução é extrair features **relativas à própria mão**:
1. Subtrair a posição do pulso de todos os pontos (centraliza)
2. Dividir pela distância pulso→base do indicador (normaliza escala)

### Como Evitamos Falsos Positivos

| Técnica | O que faz |
|---------|-----------|
| **Confiança mínima 75%** | Ignora predições incertas |
| **Buffer de 8 frames** | Exige que o gesto apareça 6 dos 8 últimos frames |
| **Cooldown de 1,5s** | Evita repetir o mesmo gesto várias vezes seguidas |
| **Ângulos de dedos** | Features extras que distinguem sinais parecidos |

---

## 🤟 Reconhecendo Sinais Reais de Libras

### Sinais Estáticos (boas letras para começar)

Os seguintes sinais do alfabeto de Libras são **estáticos** (sem movimento) e funcionam bem com este sistema:

| Sinal | Descrição dos dedos |
|-------|---------------------|
| **A** | Punho fechado, polegar ao lado |
| **B** | 4 dedos juntos e levantados, polegar dobrado |
| **C** | Mão em formato de "C" aberto |
| **L** | Indicador e polegar formando "L" |
| **V** | Indicador e médio levantados (sinal de paz) |
| **OI** | Mínimo levantado + polegar levantado |

### Sinais com Movimento (mais avançados)

Para sinais que envolvem **movimento** (como OBRIGADO, COMO VAI VOCÊ), você precisará de uma estratégia diferente: gravar sequências de landmarks ao longo do tempo.

**Abordagem futura:** usar LSTM (rede neural recorrente) que aprende padrões temporais. Isso pode ser o próximo passo após o TCC.

---

## 💡 Como Melhorar a Precisão

### Quantidade de dados
- **Mínimo aceitável:** 100 amostras por gesto
- **Recomendado:** 200–500 amostras por gesto
- **Ideal:** coletar de 3 pessoas diferentes

### Diversidade de dados
Colete em condições variadas:
- Diferentes fundos (branco, colorido, escuro)
- Diferentes iluminações
- Diferentes posições da mão no frame
- Com e sem manga longa

### Escolha do algoritmo
| Situação | Algoritmo recomendado |
|----------|-----------------------|
| Poucos dados (< 300 total) | **KNN** |
| Muitos dados (> 300 total) | **SVM** |
| Você quer experimentar | Teste os dois e compare |

---

## 🔮 Futuramente: Transformar em Luva Inteligente Offline

Esta é uma excelente proposta de trabalho futuro para o TCC:

### Arquitetura da Luva

```
Sensores na Luva
│
├── 5 Flex Sensors (um por dedo)
│   └── Mede quanto cada dedo está dobrado (0–90°)
│
├── IMU (giroscópio + acelerômetro)
│   └── Detecta orientação e movimento da mão
│
└── Microcontrolador (ESP32 ou Arduino Nano BLE)
    └── Lê os sensores → envia por Bluetooth
        ↓
    Aplicativo no celular ou computador
    (usa o mesmo classificador treinado aqui)
```

### Por que é viável?

O modelo treinado neste projeto usa **vetores de números** (ângulos, posições). Os sensores da luva também produzem números. Você pode **reaproveitar o mesmo classificador** — apenas mudando de onde vêm os dados de entrada.

### Materiais necessários
- Flex sensors × 5 (cerca de R$ 15 cada)
- MPU-6050 (IMU) ≈ R$ 15
- ESP32 ≈ R$ 40
- Luva de lycra para costurar os sensores

### Vantagem
Funciona **sem câmera e sem internet** — ideal para uso cotidiano por pessoas surdas.

---

## 📊 Métricas para o TCC

Quando executar `treinar_modelo.py`, você receberá:

- **Acurácia** — % de gestos reconhecidos corretamente
- **Precisão** — quando diz "A", quantas vezes realmente é "A"
- **Recall** — de todos os "A" reais, quantos foram detectados
- **F1-Score** — média harmônica entre precisão e recall
- **Validação cruzada (5-fold)** — resultado mais confiável que uma única divisão

Esses números devem ir na seção de "Avaliação de Resultados" do TCC.

---

## 📚 Referências Sugeridas para o TCC

- Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. CVPR Workshop.
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3).
- Capovilla, F. C. & Raphael, W. D. (2001). *Dicionário Enciclopédico Ilustrado Trilíngue da Língua de Sinais Brasileira.*

---

*Sistema desenvolvido como TCC — adaptado para fins educacionais.*
