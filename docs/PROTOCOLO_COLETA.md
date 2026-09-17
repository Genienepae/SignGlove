# Protocolo de coleta do SignGlove

Use este roteiro apenas após combinar a atividade com o professor orientador e
com as pessoas participantes. O objetivo é obter dados organizados para avaliar
o protótipo; não é testar proficiência em Libras.

## Antes de começar

1. Peça autorização para participar e explique que o sistema registra apenas
   landmarks da mão e códigos anônimos, sem salvar vídeo, foto ou nome.
2. Crie um código por pessoa: `P01`, `P02`, `P03` e assim por diante. Não use
   nome, telefone, e-mail ou matrícula no aplicativo.
3. Defina com a consultora de Libras quais configurações de mão serão usadas.
   O conjunto atual contém A, B, C, D e F; não inclua novas letras sem revisão.
4. Posicione a câmera em altura estável, com a mão inteira visível e fundo sem
   muito movimento. Anote iluminação e equipamento em uma planilha separada
   autorizada pelo orientador.

## Para cada participante

1. Informe o código da pessoa no campo **CÓDIGO DO PARTICIPANTE**.
2. Grave cada gesto em uma sessão separada. Faça pequenas mudanças naturais de
   distância e orientação, sem mudar a configuração que está sendo ensinada.
3. Registre a mesma lista de gestos para cada pessoa. Isso é necessário para a
   validação por participante funcionar.
4. Faça pausas curtas entre gestos e, se possível, colete uma segunda sessão em
   outro dia ou com outra iluminação. Anote essa condição fora do aplicativo.
5. Não misture duas pessoas com o mesmo código e não reutilize um código em uma
   pessoa diferente.

## Quantidade inicial sugerida

Para um piloto, use pelo menos 3 participantes e 2 sessões por participante.
Em cada sessão, grave 50 a 80 amostras de cada gesto. Mais frames não substituem
mais pessoas: a prioridade é repetir os mesmos gestos com participantes distintos.

## Depois da coleta

1. Confira se `data/metadata/coletas.jsonl` foi criado. Execute
   `python training/verificar_cobertura.py` para ver a tabela de letras por
   participante e localizar uma coleta faltante.
2. Treine a IA. Se houver todos os gestos em pelo menos duas pessoas, a interface
   mostrará **acurácia por participante**.
3. Execute `python training/avaliar_por_participante.py` para registrar acurácia
   e F1 macro sem misturar pessoas entre treino e teste.
4. Anote falhas observadas por gesto, mão não detectada, iluminação e tempo de
   resposta. Não descarte resultados ruins sem registrar o motivo.

## Prática guiada

O modo desafio gera `data/metadata/praticas.jsonl` com duração, acertos e erros.
Esses dados mostram uso do protótipo, mas não provam aprendizagem. Para avaliar
aprendizagem, combine com o orientador uma atividade antes e depois da prática,
com critérios humanos revisados por alguém qualificado em Libras.
