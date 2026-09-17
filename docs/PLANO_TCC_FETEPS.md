# SignGlove: diagnóstico e proposta para o TCC

Revisão em 16/09/2026, a partir do commit `05ef86f`. Contexto: TCC de
Desenvolvimento de Sistemas da Etec de Registro, previsto para o próximo ano,
com interesse em participar da FETEPS. O estudante tem uma tia formada em Libras;
a participação dela ainda precisa ser combinada.

## Parecer

Há uma base técnica aproveitável para um TCC, mas o estado observado é um protótipo
inicial de reconhecimento pela câmera. Não há evidências suficientes para afirmar
que já compete por um top 3 ou estimar uma probabilidade de premiação. Isso depende
do trabalho concluído, dos concorrentes e das regras da edição.

Minha recomendação é investigar uma **ferramenta de prática supervisionada de
configurações de mão**, com orientações revisadas por alguém com formação em Libras
e avaliação do ganho de aprendizagem. É uma proposta a construir e testar.

## Evidências no repositório

| Item | Estado observado | Consequência |
| --- | --- | --- |
| Dados | A: 350; B, C, D e F: 150 amostras cada; total de 950 vetores de 73 características | Cinco classes não demonstram cobertura do alfabeto ou de conversação. |
| Procedência | JSONs sem identificador de pessoa ou sessão | Não permitem comprovar separação de pessoas entre treino e teste. |
| Aplicação | `main.py` usa uma mão e o classificador estático | Os scripts LSTM não estão integrados a essa aplicação. |
| Avaliação estática | `training/treinar_modelo.py` usa grupos quando há manifesto completo e avisa quando não há | Dados antigos sem participante continuam inadequados para afirmar generalização. |
| Avaliação visual | `treinar_visual.py` seleciona o modelo por grupos quando a cobertura permite | Sem lotes completos por participante, a tela mantém a métrica por amostra. |
| Avaliação LSTM | Sequências aumentadas antes da divisão interna de treino e validação | Mesmo risco de compartilhar versões de uma sequência. |
| Persistência | Interface salva `modelo` e `le`; classificador principal espera `modelo`, `label_encoder` e configurações | Unificar formatos antes de alternar entre os caminhos de treinamento. |
| Confirmação | Uma predição podia herdar a confirmação da classe anterior | Corrigido nesta revisão, com testes de troca de classe e limites da janela. |
| Luva | Sem firmware ou integração com sensores no repositório | É uma proposta futura, não um recurso implementado. |

A revisão não mediu acurácia com novos participantes, latência de webcam ou
qualidade dos sinais. O arquivo de modelo existente não foi desserializado.
As demais pendências técnicas da tabela continuam abertas.

## Feira e projetos semelhantes

Uma [luva tradutora de Libras ficou em 10º na FETEPS de 2025](https://www.cps.sp.gov.br/projeto-da-etec-contribui-para-inclusao-de-pessoas-com-deficiencia-auditiva/).
O [SCAN HANDS](https://feteps.cps.sp.gov.br/projetos/scan-hands-visao-i-a-para-libras/)
já descreve ensino e prática com câmera e feedback. O
[RoboLibras](https://biofatecou.fatecourinhos.edu.br/robolibras/index.html)
também apresenta prática com reconhecimento por IA. Usar câmera, luva ou uma tela
de exercícios não comprova originalidade por si só.

Comparar demonstrações, publicações e funcionalidades antes de alegar um diferencial.
Uma descrição pública resumida não prova a ausência de um recurso nos concorrentes.

O [regulamento da 16ª FETEPS](https://storagefeteps.blob.core.windows.net/blobfeteps/2025/04/REGULAMENTO-16a-FETEPS-11.03.pdf)
incluiu inovação, aplicabilidade, resolução do problema, potencial de transformação,
tecnologia, apresentação e aspectos de viabilidade de mercado. Minha sugestão é
preparar evidências nessas frentes:

| Frente | Evidência a produzir |
| --- | --- |
| Problema | Dificuldade de prática confirmada por aprendizes e consultores em Libras. |
| Diferencial | Comparação concreta com ferramentas semelhantes e estudo por vídeo. |
| Aplicabilidade | Piloto com uma turma ou oficina parceira. |
| Tecnologia | Resultado em pessoas novas, análise de erros e tempo de resposta. |
| Transformação | Aprendizagem e retenção medidas com avaliação humana. |
| Viabilidade | Equipamentos, custo medido, manutenção e interesse de uma escola. |
| Apresentação | Demonstração reproduzível, relatório, limites e pitch. |

O [site oficial](https://feteps.cps.sp.gov.br/) anuncia a 17ª edição com novidades
em breve. Usar 2025 apenas como referência; conferir datas, equipe e documentos da
edição pretendida com o orientador quando publicados.

A [notícia dos premiados de 2025](https://www.cps.sp.gov.br/grande-vencedor-da-16a-feteps-e-da-etec-de-ribeirao-pires/)
lista dois projetos da Etec de Registro nas posições 7 e 9. Conversar com seus
orientadores pode ajudar na organização da pesquisa e da apresentação.

## Recorte proposto

**Título provisório:** SignGlove — apoio à prática de configurações de mão com
feedback visual e avaliação de aprendizagem.

**Público inicial:** aprendizes ouvintes em uma oficina supervisionada de Libras.

**Pergunta de pesquisa:** a prática com feedback visual específico melhora o
desempenho e a retenção de um conjunto delimitado de configurações de mão, comparada
ao estudo dos mesmos exemplos em vídeo pelo mesmo tempo?

Começar com 5 a 10 configurações estáticas escolhidas após revisão do conteúdo.
Apresentar referência, observar a tentativa, oferecer orientação verificável e
permitir nova tentativa. Uma indicação sobre a flexão de um dedo só deve aparecer
com regra validada e pontos suficientemente confiáveis. Quando a câmera não
permitir avaliar, pedir reposicionamento.

A classe prevista e a confiança do modelo atual **não são uma explicação pedagógica
nem uma nota de correção linguística**. O feedback exige implementação própria,
referências adequadas e comparação com avaliação humana. Evitar transformações dos
dados que eliminem distinções relevantes entre sinais.

Uma mão isolada não representa todos os componentes da Libras. O
[material de Letras Libras da UFSC](https://www.libras.ufsc.br/colecaoLetrasLibras/eixoFormacaoBasica/foneticaEFonologia/scos/cap15009/5.html)
descreve configuração, movimento, localização, orientação e componentes não manuais.
Apresentar o sistema como prática de conteúdo delimitado, com orientação humana.

## Participação da tia e de outras pessoas

A tia pode ajudar a selecionar conteúdo, revisar referências, identificar erros
frequentes e avaliar as orientações do sistema, conforme sua formação e
disponibilidade. Combinar com ela quais tarefas consegue assumir.

Incluir pessoas surdas na definição do problema e na avaliação; uma única consultora
não representa todas as experiências e variantes da língua. O professor orientador
deve acompanhar metodologia, documentação e procedimentos de participação antes da coleta.

## Avaliação necessária

1. Registrar códigos de participante, sessão e tentativa, classe, mão e iluminação.
   Use códigos como `P01/S01` e `P01/S02`; centenas de frames seguidos não equivalem a centenas de demonstrações independentes.
   Não inventar metadados para os JSONs antigos.
2. Reservar pessoas inteiras para teste. Separar sessões e tentativas conforme a
   pergunta experimental. A documentação do
   [scikit-learn sobre grupos](https://scikit-learn.org/1.5/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
   explica a importância desse procedimento para amostras do mesmo indivíduo.
3. Gerar variações apenas no treino, depois da separação e dentro de cada divisão
   da validação cruzada. Escolher limites de confiança sem consultar o teste final.
4. Relatar precisão, recall, F1 por classe, matriz de confusão e resultado por pessoa.
   Testar transições e gestos fora do vocabulário: confiança alta não comprova
   que uma entrada seja conhecida.
5. Comparar as orientações de feedback com avaliação humana e contar sugestões
   incorretas. Acurácia de letras não valida a qualidade da orientação.
6. Avaliar antes, depois e uma semana após a prática. Comparar com vídeos de mesma
   duração; distribuir participantes entre condições de forma previamente definida,
   preferencialmente aleatória. Quando viável, o avaliador não deve saber qual
   ferramenta cada aprendiz utilizou.
7. Informar tamanho da amostra, falhas, equipamentos, tempo de resposta e incerteza.
   Definir os grupos com o orientador e começar com um piloto. Um estudo pequeno
   informa viabilidade, sem comprovar eficácia para toda a população.

## Próximas entregas

| Período relativo | Entrega |
| --- | --- |
| Semanas 1–2 | Reunião com tia e orientador; parceiro; problema confirmado com exemplos. |
| Semanas 3–4 | Comparação de soluções; referências revisadas; protocolo de coleta e avaliação. |
| Mês 2 | Formato único de modelos, coleta com metadados e avaliação sem mistura entre conjuntos. |
| Mês 3 | Feedback para poucos erros revisados e medição de concordância com avaliação humana. |
| Meses 4–5 | Piloto, correções e estudo de aprendizagem. |
| Mês 6 em diante | Relatório com dados reais, custos, vídeo, apresentação e adequação ao edital. |

Adaptar às aulas e ao edital; esse calendário não estabelece datas da FETEPS.
Expandir para sinais dinâmicos, duas mãos ou hardware se a investigação justificar
e houver tempo de testar. Não há resultado pedagógico comprovado nesta revisão.

## Alternativa para comparar

Se as entrevistas não confirmarem uma necessidade de prática, investigar um
**sistema de acompanhamento de lotes de banana para priorizar venda e reduzir
descarte**, com registros simples e, se útil, fotos padronizadas de maturação.
A relação da bananicultura com o Vale do Ribeira e os cuidados de pós-colheita
são tratados no [manual técnico da CATI](https://www.cati.sp.gov.br/portal/themes/unify/arquivos/produtos-e-servicos/acervo-tecnico/producao_vegetal/Manual_tecnico_82_Cultivo_da_Bananeira.pdf).

Essa alternativa precisa de produtor ou comerciante parceiro que confirme o
problema e permita medir descarte por lote. Cor da casca não comprova segurança
alimentar. Redução de perdas, utilidade da IA e originalidade exigem investigação
própria. Com o código existente e a possível colaboração da tia, faz sentido testar
primeiro a hipótese do SignGlove.
