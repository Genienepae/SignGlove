# SignGlove

[![Testes](https://github.com/Genienepae/SignGlove/actions/workflows/testes.yml/badge.svg)](https://github.com/Genienepae/SignGlove/actions/workflows/testes.yml)

Protótipo de reconhecimento de configurações de mão pela webcam, desenvolvido
como base para um TCC de Desenvolvimento de Sistemas.

O código atual usa Python, MediaPipe e classificadores de aprendizado de máquina.
Os dados versionados contêm 950 amostras das letras **A, B, C, D e F**. A coleta
de outras classes e o reconhecimento de sequências são possibilidades de evolução;
a presença desses scripts não demonstra que esses recursos já foram validados.

## Começar

- [Instalação, execução e limitações técnicas](libras_tcc/README.md).
- [Diagnóstico e proposta de evolução para o TCC e a FETEPS](docs/PLANO_TCC_FETEPS.md).
- [Protocolo de coleta com participantes](docs/PROTOCOLO_COLETA.md).
- [Roteiro de demonstração para a banca](docs/ROTEIRO_DEMONSTRACAO.md).
- [Formulário para o piloto de prática](docs/FORMULARIO_PILOTO.md).
- [Diário de experimentos para o TCC](docs/DIARIO_EXPERIMENTOS.md).
- [Checklist de apresentação](docs/CHECKLIST_APRESENTACAO.md).

## Escopo

Esta versão reconhece classes cadastradas a partir de uma mão. Não é um tradutor
de conversas em Libras. A língua também envolve movimento, localização,
orientação, expressões faciais e corporais, conforme o
[material de Letras Libras da UFSC](https://www.libras.ufsc.br/colecaoLetrasLibras/eixoFormacaoBasica/foneticaEFonologia/scos/cap15009/5.html).

A luva com sensores é uma proposta futura: não há firmware nem integração com
sensores neste repositório. Um modelo treinado com landmarks de câmera precisa
de uma nova representação de entrada e de treinamento adequado para dados de luva.

O caminho proposto para o TCC é investigar uma ferramenta de prática supervisionada
de um conjunto delimitado de configurações de mão, com conteúdo revisado por
especialistas em Libras e avaliação com participantes. Essa proposta ainda precisa
ser construída e testada.

## Testes automatizados

Com NumPy e scikit-learn instalados, execute na raiz do repositório:

```powershell
python -m unittest discover -s tests -v
```

O CI usa as mesmas versões mínimas definidas em
`libras_tcc/requirements-test.txt` e não precisa acessar a câmera.

Os testes verificam a confirmação temporal de gestos, inclusive a troca entre
classes. Eles não medem a precisão do reconhecimento por câmera nem a eficácia
pedagógica da proposta.

## Referência de Libras

O treinador tem um botão para abrir o [Alfabeto de Libras e Configuração de
Mãos do INES](https://www.gov.br/ines/pt-br/central-de-conteudos/publicacoes-1/todas-as-publicacoes/alfabeto-manual-e-configuracao-de-maos)
em uma janela ao lado da prática. O material é publicado pelo INES sob CC BY-ND
3.0; ele é aberto na fonte oficial, sem cópia ou alteração dentro do repositório.
