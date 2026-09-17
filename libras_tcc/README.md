# 🤟 Sistema de Reconhecimento de Libras com IA

**Protótipo para TCC — reconhecimento de configurações de mão usando MediaPipe e Machine Learning**

O conjunto versionado tem 950 amostras de cinco letras: A, B, C, D e F.
Esta versão usa uma mão pela câmera; não traduz conversas em Libras. A luva é
uma proposta futura. Consulte o [plano para o TCC e a FETEPS](../docs/PLANO_TCC_FETEPS.md)
para o diagnóstico e a proposta de pesquisa.

As métricas dos scripts atuais são exploratórias: faltam metadados para separar
pessoas e sessões entre treino e teste. A interface visual e o treinamento LSTM
também precisam separar os dados **antes** de gerar variações de amostras.

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

Antes de uma demonstração, confira o estado local sem abrir câmera:

```bash
python training/diagnosticar_projeto.py
```

Ele informa quantidade de amostras, modelo salvo e se a avaliação por participante
já pode ser feita.

### 1. Instalar dependências

Use Python 3.12 com as versões fixadas neste projeto. A partir da raiz do
repositório, crie e ative um ambiente isolado antes de executar os comandos abaixo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd libras_tcc
```

```bash
python -m pip install -r requirements.txt
```

### 2. Coletar amostras de cada gesto
```bash
# Coleta 200 amostras da letra A
python training/coletar_dados.py --gesto A --amostras 200 --participante P01 --sessao S01

# Repita para cada gesto que quiser reconhecer
python training/coletar_dados.py --gesto B --amostras 200 --participante P01 --sessao S01
python training/coletar_dados.py --gesto C --amostras 200 --participante P01 --sessao S01
python training/coletar_dados.py --gesto D --amostras 200 --participante P01 --sessao S01
```
> **Dica:** Pressione ESPAÇO para iniciar/pausar a coleta dentro do programa.

### 3. Treinar o modelo
```bash
python training/treinar_modelo.py
```

Para salvar a métrica principal em JSON:

```powershell
python training/treinar_modelo.py --saida reports/treino.json
```
Isso exibe acurácia, F1 macro, precisão e recall por gesto. O arquivo JSON salva
essas métricas por letra, a matriz de confusão, o critério de avaliação, gestos e quantidade de amostras. Com
os dados atuais, são estimativas exploratórias por amostra, não uma avaliação em
pessoas novas.

Como A tem mais amostras que as outras letras atuais, a SVM é treinada com pesos
balanceados por classe. Isso reduz o viés para a letra mais frequente; confirme o
efeito com F1 macro e avaliação por participante depois de uma nova coleta.

### Treinador visual

```bash
python treinar_visual.py
```

Para usar a câmera do celular quando ela aparecer como outro dispositivo:

```bash
python treinar_visual.py --camera 1
```

Para validar o modelo visual sem abrir janela ou câmera:

```powershell
python treinar_visual.py --check
```

O treinador exibe a webcam em até 20 FPS para manter a janela responsiva e limita
o processamento pesado do detector a essa mesma taxa. Ele usa
o modelo leve do MediaPipe. Ao clicar em **ABRIR ALFABETO OFICIAL**, abre a
publicação de referência do INES no navegador para consulta lado a lado. Depois
de coletar os gestos, o programa compara SVM e floresta aleatória com validação
cruzada e usa o melhor candidato para aquele conjunto.
No modo teste, previsões abaixo de 70% são tratadas como dúvida e não entram na
confirmação temporal.

A referência é material de consulta; valide a seleção de sinais e qualquer
orientação de prática com uma pessoa formada em Libras.

Quando o manifesto de coleta tiver os mesmos gestos registrados para pelo menos dois
participantes, a seleção usa divisões por participante e a tela mostra **acurácia por
participante**. Sem essa cobertura, mostra **acurácia por amostra**, que não mede
pessoas novas.

### Coleta com participantes

Antes de gravar, informe um **código anônimo** de participante, como `P01` ou
`P02`; não use nome completo. Cada gravação nova gera um registro em
`data/metadata/coletas.jsonl` com o código, gesto, horário e intervalo das
amostras daquele lote. Use também sessões como `S01` e `S02` para separar dias
ou condições de coleta. Esses registros permitem reservar participantes inteiros
para testes futuros. As amostras antigas não têm esse histórico e não devem ser
apresentadas como coleta por participante.

Depois de coletar pelo menos dois participantes, execute:

```bash
python training/avaliar_por_participante.py
```

Para guardar o resultado daquele experimento em um arquivo:

```bash
python training/avaliar_por_participante.py --saida reports/avaliacao_piloto.json
```

O comando mede acurácia e F1 macro reservando pessoas inteiras para teste. Para
um resultado mais estável, colete os mesmos gestos com pelo menos três pessoas.
Ele também mostra precisão, recall, F1 e quantidade de amostras por gesto, além
da matriz de confusão para identificar quais letras foram confundidas.

Antes disso, confira se todos gravaram todos os gestos:

```bash
python training/verificar_cobertura.py
```

### Modo desafio

Depois de treinar a IA, clique em **INICIAR DESAFIO**. O sistema abre o PDF de
referência, sorteia uma das letras treinadas e contabiliza o acerto apenas após a
confirmação temporal da câmera. Também informa erros estáveis, sem repetir o mesmo
erro continuamente. A pontuação serve para acompanhar a prática na
sessão; não é uma avaliação linguística nem uma medida de aprendizagem validada.
Ao encerrar o desafio, o sistema salva em `data/metadata/praticas.jsonl` somente
o código anônimo do participante, início, duração, acertos e erros. Não salva imagens,
vídeos ou nomes.

Para resumir as sessões já registradas, sem abrir a câmera:

```bash
python training/resumo_praticas.py
```

Para guardar o resumo em JSON e anexar as métricas ao relatório:

```powershell
python training/resumo_praticas.py --saida reports/resumo_praticas.json
```

O resumo mostra também a taxa de acerto agregada por participante. Ele descreve o
uso do protótipo e não comprova ganho de aprendizagem sem
um protocolo de avaliação definido com o orientador.

### 4. Executar o reconhecimento em tempo real
```bash
python main.py
```

---

O reconhecimento principal usa câmera em 640×480 e o modelo leve do MediaPipe
para diminuir atraso em computadores escolares. O OpenCV também fica limitado a
um thread para evitar disputa de CPU. Se a câmera externa estiver sendo
usada por outro programa, feche esse programa antes de abrir o reconhecimento.
Se a câmera do celular aparecer como outro dispositivo, experimente:

```bash
python main.py --camera 1
```

Para verificar o modelo sem abrir nenhuma câmera:

```bash
python main.py --check
```

Antes de uma coleta ou treino, o diagnóstico também pode ser usado em modo
estrito. Ele retorna código de erro quando faltam amostras ou há vetores
inválidos, permitindo interromper um script automaticamente:

```powershell
python training/diagnosticar_projeto.py --strict
```

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

Revise as referências com uma pessoa com formação em Libras antes de coletar.
Uma configuração de dedos isolada não basta para definir qualquer sinal.

### Sinais com Movimento (mais avançados)

Para sinais que envolvem **movimento** (como OBRIGADO, COMO VAI VOCÊ), você precisará de uma estratégia diferente: gravar sequências de landmarks ao longo do tempo.

**Abordagem futura:** usar LSTM (rede neural recorrente) que aprende padrões temporais. Isso pode ser o próximo passo após o TCC.

---

## 💡 Como Melhorar a Precisão

### Quantidade de dados
Não há uma quantidade universal de frames que garanta boa precisão. Priorize
tentativas independentes de diferentes pessoas e sessões, registre a procedência
e reserve participantes para teste. Frames consecutivos são correlacionados.

### Diversidade de dados
Colete em condições variadas:
- Diferentes fundos (branco, colorido, escuro)
- Diferentes iluminações
- Diferentes posições da mão no frame
- Com e sem manga longa

### Escolha do algoritmo
| Situação | Algoritmo recomendado |
|----------|-----------------------|
| Modelo de referência | Compare KNN e SVM com as mesmas divisões de avaliação |
| Escolha final | Use resultados de validação por pessoa e custo de execução |

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
    (usa um novo modelo treinado com dados dos sensores)
```

### Por que é viável?

O algoritmo de classificação pode ser reutilizado como abordagem, mas o modelo
treinado com os 73 valores de landmarks não aceita automaticamente leituras de
sensores. A luva exige outra representação, calibração, coleta e treinamento com
dados próprios. O repositório ainda não implementa essa integração.

### Materiais necessários
- Flex sensors × 5
- MPU-6050 (IMU)
- ESP32
- Luva de lycra para costurar os sensores

Definir a arquitetura e obter cotações atuais antes de estabelecer um orçamento.

### Vantagem
O objetivo seria operar sem câmera e sem internet. A viabilidade, o conforto e a
utilidade para o público precisam de testes com um protótipo físico.

---

## 📊 Métricas para o TCC

Quando executar `treinar_modelo.py`, você receberá:

- **Acurácia** — % de gestos reconhecidos corretamente
- **Precisão** — quando diz "A", quantas vezes realmente é "A"
- **Recall** — de todos os "A" reais, quantos foram detectados
- **F1-Score** — média harmônica entre precisão e recall
- **Validação cruzada (5-fold)** — estimativa por amostra; com dados correlacionados,
  é necessário separar por pessoa ou sessão para medir generalização

Ao apresentar esses números no TCC, informe o protocolo e suas limitações. Não
trate a avaliação atual como comprovação de funcionamento com novos usuários.

---

## 📚 Referências Sugeridas para o TCC

- Zhang, F. et al. (2020). *MediaPipe Hands: On-device Real-time Hand Tracking*. CVPR Workshop.
- Cortes, C. & Vapnik, V. (1995). *Support-vector networks*. Machine Learning, 20(3).
- Capovilla, F. C. & Raphael, W. D. (2001). *Dicionário Enciclopédico Ilustrado Trilíngue da Língua de Sinais Brasileira.*

---

*Sistema desenvolvido como TCC — adaptado para fins educacionais.*
