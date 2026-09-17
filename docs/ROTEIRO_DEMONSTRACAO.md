# Roteiro de demonstração do SignGlove

Este roteiro apresenta o estado real do protótipo. Ajuste o tempo à regra da
feira e não afirme que o sistema traduz conversas em Libras.

## Preparação antes da banca

1. Execute `python training/diagnosticar_projeto.py` e confira que o modelo foi
   encontrado.
2. Feche programas que usam a câmera do celular.
3. Abra o Treinador Visual ou o reconhecimento principal. Se necessário, use
   `python main.py --camera 1`.
4. Deixe o PDF oficial do INES disponível como referência visual.
5. Tenha uma pessoa que conheça os sinais revisando a demonstração, quando
   possível. Não improvise letras que não estão no modelo.

## Demonstração de 3 minutos

### 0:00 a 0:30 — Problema e recorte

“Aprender configurações de mão exige prática e retorno imediato. O SignGlove é
um protótipo de prática para um conjunto delimitado de configurações de mão,
não um tradutor de Libras.”

Mostre as cinco classes atuais: A, B, C, D e F.

### 0:30 a 1:20 — Tecnologia

“A câmera detecta 21 pontos da mão com MediaPipe. O sistema extrai 73
características e uma IA escolhe entre SVM e floresta aleatória. A confirmação
temporal exige estabilidade antes de aceitar uma letra.”

Mostre uma letra sendo reconhecida e, em seguida, mude a mão para que a banca
veja que a confirmação não é imediata.

### 1:20 a 2:00 — Prática

Abra o **Modo Desafio**. Mostre a letra solicitada, faça um acerto e mostre que
o placar registra acertos e erros estáveis. Explique que o PDF oficial aberto
ao lado serve de referência durante a tentativa.

### 2:00 a 2:40 — Evidências

“O sistema registra participantes por códigos anônimos, por exemplo P01, e
sessões como S01. Quando houver os mesmos gestos em pessoas diferentes, a
avaliação separa participantes inteiros entre treino e teste e gera precisão,
recall, F1 e matriz de confusão por letra.”

Mostre `python training/verificar_cobertura.py` ou um JSON de avaliação já
gerado, caso existam dados reais.

### 2:40 a 3:00 — Limites e próximos passos

“Hoje o sistema reconhece cinco configurações estáticas de uma mão. Libras
também envolve movimento, localização, orientação e componentes não manuais.
O próximo passo é coletar dados com participantes, validar as referências com
pessoas qualificadas em Libras e medir a prática em um piloto.”

## Perguntas que podem aparecer

| Pergunta | Resposta honesta |
| --- | --- |
| Isso traduz Libras? | Não. O protótipo pratica cinco configurações estáticas de mão. |
| A IA foi testada com outras pessoas? | A estrutura está pronta, mas o resultado só deve ser apresentado depois da coleta identificada por participante. |
| Por que usar códigos? | Para organizar a avaliação sem usar nomes ou imagens dos participantes. |
| Qual é o diferencial? | Juntar referência visual, reconhecimento local, desafio e avaliação por participante em uma prática delimitada e revisável. |
