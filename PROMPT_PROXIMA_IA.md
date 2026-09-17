# Prompt para a próxima IA

Estou desenvolvendo o TCC **SignGlove**, da ETEC de Registro (DS, 2º ano), para
apresentar na FETEPS. O repositório é `https://github.com/Genienepae/SignGlove`.
Trabalhe em `C:\Users\gabriel.silva\SignGlove` e fale em português simples.

## Situação atual

- Branch: `main`.
- Último commit publicado: `a665d81 Exibe alfabeto durante gravacao de gestos`.
- O programa principal de treino é `libras_tcc/treinar_visual.py`.
- Não abra a câmera ou a interface por conta própria: o usuário usa a câmera do
  celular pelo Windows e pode travar. Use somente verificações sem câmera,
  como `--check`, salvo se ele pedir para abrir.
- A imagem oficial do alfabeto do INES está em
  `libras_tcc/assets/referencias/alfabeto_libras_ines.png`. Ela aparece dentro
  da mesma janela, ao lado da câmera, durante teste, desafio e gravação.
- O treinador abre maximizado para a câmera não ficar pequena.
- O detector desenha até duas mãos. A mão direita recebe dedos 1–5 e a esquerda
  6–10; cada dedo usa uma cor própria. O modelo atual ainda classifica somente
  a mão principal, porque usa 73 features de uma mão.
- Foram corrigidos os falsos avisos de “mão não detectada” entre as leituras e
  o piscar das letras no modo de teste.
- O modelo salvo reconhece apenas `A`, `B`, `C`, `D` e `F`. Não diga que ele
  reconhece o alfabeto completo.
- Há dados locais que **não devem ser enviados sem confirmação**:
  `libras_tcc/data/gestures/A.json`, `libras_tcc/models/modelo_libras.pkl` e
  `libras_tcc/data/metadata/`.

## Pedido pendente do usuário

Ele pediu: “procure vídeos e os use para treinar a IA”. Já foram encontradas
fontes úteis:

- INES: https://debasi.ines.gov.br/atividades-diversas/jogos-e-atividades
  tem vídeo de “Alfabeto em Libras: configuração de mão”.
- Dicionário do INES: https://dicionario.ines.gov.br/ possui vídeos por sinal.
- Há conjuntos acadêmicos, como Libras91, mas antes de baixar ou usar qualquer
  vídeo/dataset é obrigatório conferir licença, forma de acesso e se o formato
  atende ao projeto.

Não use vídeos aleatórios do YouTube como dados de treino e não baixe conteúdo
sem licença clara. Um vídeo de referência não é dataset rotulado suficiente.
Para reconhecer sinais com duas mãos ou movimento, será necessário mudar a
representação atual e coletar/usar sequências de landmarks, depois treinar e
avaliar um modelo próprio. A tia do usuário é formada em Libras e deve revisar
os sinais e o protocolo de coleta.

## Próximo passo recomendado

Explique isso de forma objetiva e proponha uma implementação séria: importar
vídeos próprios ou um dataset com licença clara, extrair landmarks por frame,
registrar a origem/licença e treinar uma avaliação separada por participante.
Não invente métricas nem alegue que um modelo foi treinado com A–Z sem dados.

## Comandos seguros

No diretório raiz:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -c "import ast,pathlib; files=list(pathlib.Path('libras_tcc').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print(f'Sintaxe AST OK: {len(files)} arquivos')"
```

No diretório `libras_tcc`:

```powershell
..\.venv\Scripts\python.exe treinar_visual.py --check
..\.venv\Scripts\python.exe main.py --check
..\.venv\Scripts\python.exe training\diagnosticar_projeto.py
```

Ao editar, use `apply_patch`, rode os testes necessários e faça commit/push
somente dos arquivos de código e documentação. Nunca inclua automaticamente os
dados de coleta ou o modelo local do usuário.
