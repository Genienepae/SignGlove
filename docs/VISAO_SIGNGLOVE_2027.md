# SignGlove 2027 — ponto de partida

Data da decisão: 22 de setembro de 2026.

## Visão do estudante

Transformar o SignGlove em um sistema discreto de comunicação em Libras. A
visão de longo prazo é uma pessoa usar uma única luva fina, uma câmera e um
aplicativo ou site capaz de converter Libras para português em texto e voz. No
sentido contrário, a fala de uma pessoa ouvinte deve virar texto para apoiar a
conversa.

No futuro, o produto pode migrar para óculos com aparência comum, câmera,
áudio aberto ou visor, acompanhados por uma pulseira sensorial. A inspiração de
formato é a combinação de óculos e pulseira dos Meta Ray-Ban Display, sem copiar
marca, hardware proprietário ou afirmar equivalência técnica.

## Resposta técnica honesta

Uma câmera ajuda a observar as duas mãos, braços, corpo e rosto, mas não
"reconhece tudo" automaticamente. Uma tradução contínua ainda precisa resolver:

- o começo e o fim de cada sinal;
- movimentos e transições entre sinais;
- configuração, orientação e localização das mãos;
- o uso simultâneo das duas mãos;
- expressão facial, boca, cabeça, ombros e direção do olhar;
- referências criadas no espaço de sinalização;
- gramática de Libras, contexto e variações regionais;
- o modo de sinalizar de pessoas que não participaram do treinamento;
- o tratamento de sinais desconhecidos sem inventar uma tradução.

Libras não é português representado gesto por gesto. Portanto, reconhecer
classes isoladas e traduzir uma conversa livre são problemas diferentes. Não
existe uma lista finita de "todas as palavras" que encerre o desenvolvimento de
um tradutor de língua natural.

## Arquitetura proposta

```text
Luva fina na mão dominante
  5 sensores de flexão + IMU + sensores de contato
                         \
Câmera RGB ----------------> sincronização multimodal
  mãos + corpo + rosto     /             |
                                        v
                          reconhecimento temporal de sinais
                                        |
                                        v
                         tradução Libras -> português
                                        |
                              texto + voz + confiança

Microfone -> reconhecimento de fala -> texto para a resposta
```

A luva mede com estabilidade aquilo que a câmera perde com iluminação,
oclusão ou dedos sobrepostos. A câmera mede a mão sem luva, a posição em
relação ao corpo e os marcadores faciais que a luva não consegue observar. O
sistema deve continuar funcionando somente com a câmera quando a luva estiver
desligada, apresentando uma confiança menor quando necessário.

Uma única luva é viável como sensor auxiliar da mão dominante. Ela não mede a
mão não dominante; esse papel fica com a câmera. Para sinais em que as duas
mãos se ocultam ou fazem contatos complexos, uma segunda luva pode ser uma
expansão futura, não uma dependência obrigatória do primeiro produto.

## Meta máxima realista para doze meses

O objetivo para a FETEPS não é alegar tradução completa da língua. É demonstrar
uma base tecnológica real, expansível e avaliada:

1. Versão web instalável e responsiva, com funcionamento offline durante a
   apresentação.
2. Reconhecimento do alfabeto manual, tratando letras com movimento como
   sequências temporais.
3. Vocabulário de 30 a 50 sinais úteis revisados por especialista em Libras.
4. Pequeno modo de conversa com frases formadas pelo vocabulário validado.
5. Libras para texto e voz; fala em português para texto.
6. Uma luva funcional com telemetria sem fio e calibração por usuário.
7. Fusão comparável em três condições: somente câmera, somente luva e híbrido.
8. Rejeição de entrada desconhecida e indicação clara de baixa confiança.
9. Avaliação com participantes inteiros fora do treino, sem misturar amostras da
   mesma pessoa entre treino e teste.
10. Relatório de acurácia, F1 macro, matriz de confusão, latência, autonomia,
    custo e falhas observadas.
11. Teste de usabilidade e revisão linguística com a tia do estudante, que é
    formada em Libras, e preferencialmente com pessoas surdas fluentes.
12. Demonstração ao vivo reproduzível, sem depender da internet.

Metas de engenharia podem ser definidas antes da coleta, como latência inferior
a um segundo e desempenho em participantes novos. Elas devem ser tratadas como
metas até serem medidas, nunca como resultados antecipados.

## Comparação com a FETEPS 2025

A Luva Tradutora de Libras da Etec Takashi Morita ficou em 10º lugar na 16ª
FETEPS, em 2025. A descrição pública informa sensores de movimento e flexão,
microcontrolador, aprendizado de máquina e saída para texto ou áudio em celular
ou tablet. O material público consultado não informa quantidade de sinais,
acurácia, matriz de confusão nem avaliação com pessoas fora do treino. Essa
ausência pública não prova que a equipe não realizou esses testes.

Para competir por top 3, o SignGlove precisa ir além da promessa e apresentar:

- protótipo físico e software integrados;
- funcionamento ao vivo;
- resultados reproduzíveis e limitações documentadas;
- participação real da comunidade atendida;
- diferencial híbrido entre câmera e sensores;
- viabilidade, custo, conforto, privacidade e caminho de mercado.

Os critérios divulgados da FETEPS incluem originalidade, inovação, viabilidade,
potencial de mercado, apresentação e impactos. A recomendação do projeto para
um investidor recebe peso especial. Top 3 é possível, mas não pode ser garantido.

## Plano de doze meses

### Meses 1 e 2 — escopo e prova de hardware

- Definir 30 a 50 sinais com a especialista em Libras.
- Construir uma luva com ESP32, cinco sensores de flexão, IMU e contatos.
- Definir protocolo versionado de telemetria, calibração e sincronização.
- Medir ruído, repetibilidade, taxa de amostragem e autonomia.

### Meses 3 e 4 — coleta multimodal

- Integrar os sensores ao treinador existente.
- Gravar câmera e sensores com relógios sincronizados.
- Registrar participante, sessão, sinal, lado dominante e consentimento.
- Criar checagem automática de cobertura e qualidade da coleta.

### Meses 5 e 6 — modelos comparáveis

- Manter um modelo simples como referência.
- Treinar modelo temporal para movimentos.
- Avaliar por participante e comparar câmera, luva e fusão.
- Adicionar classe desconhecida e limiar de confiança.

### Meses 7 e 8 — conversa e produto

- Montar o fluxo Libras para texto/voz.
- Adicionar fala para texto na direção inversa.
- Criar modo offline, histórico local e controles de privacidade.
- Melhorar conforto, acabamento e manutenção da luva.

### Meses 9 e 10 — validação externa

- Recrutar participantes que não apareceram no treino.
- Executar protocolo com tarefas iguais para todos.
- Medir desempenho, latência, erros, conforto e intenção de uso.
- Corrigir as classes mais confundidas sem esconder resultados ruins.

### Meses 11 e 12 — FETEPS

- Congelar uma versão estável para demonstração.
- Preparar pôster, artigo, vídeo, pitch e diário de experimentos.
- Ensaiar demonstração normal, sem internet e com falha simulada.
- Mostrar problema, evidência, solução, impacto e expansão futura.

## Caminho posterior ao TCC

Traduzir conversas livres exigirá um corpus muito maior, múltiplos sinalizantes
fluidos, anotações linguísticas, consentimento e uma equipe com pessoas surdas,
especialistas em Libras, software, eletrônica, IA e design de produto. A evolução
esperada é:

```text
alfabeto -> sinais isolados -> frases controladas -> sinais contínuos
         -> tradução contextual -> personalização -> produto cotidiano
```

Óculos discretos são uma possível interface futura. Para um protótipo escolar,
é mais realista usar uma câmera pequena, processamento no celular e áudio aberto
do que fabricar uma lente com projeção. Toda câmera vestível deve indicar quando
está ativa e ser usada com consentimento.

## Fontes iniciais

- FETEPS — Luva Tradutora de Libras:
  https://feteps.cps.sp.gov.br/projetos/luva-tradutora-de-libras/
- Resultado da 16ª FETEPS de 2025:
  https://alunos.no.abc.br/index.php/blog/grande-vencedor-da-16-feteps-2025-e-da-etec-de-ribeirao-pires
- UTFPR — luva com cinco sensores flexíveis, dois contatos e sensor inercial:
  https://repositorio.utfpr.edu.br/jspui/handle/1/5018
- Artigo aberto sobre luva multimodal com flexão, IMU e pressão:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13469689/
- Meta Ray-Ban Display e pulseira EMG como referência de formato:
  https://about.fb.com/br/news/2025/09/apresentando-o-meta-ray-ban-display-uma-categoria-revolucionaria-de-oculos-inteligentes/
- Plataforma de desenvolvimento de wearables da Meta:
  https://developers.meta.com/wearables/

## Regra para as próximas IAs

Trate a tradução completa de Libras como visão de longo prazo. Não afirme que a
câmera, a luva ou o modelo atual já traduzem a língua inteira. Cada ampliação de
vocabulário precisa de dados licenciados ou coletados com consentimento,
validação linguística e avaliação com participantes fora do treino.
