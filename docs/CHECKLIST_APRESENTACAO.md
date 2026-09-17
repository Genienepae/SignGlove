# Checklist de apresentação

## Antes de sair para a banca

- [ ] Confirmar que o commit apresentado está anotado no Diário de Experimentos.
- [ ] Executar `python -m unittest discover -s tests -q`.
- [ ] Executar `python training/diagnosticar_projeto.py --strict`.
- [ ] Executar `python main.py --check` para validar o modelo sem câmera.
- [ ] Executar `python treinar_visual.py --check` para validar o treinador visual sem janela.
- [ ] Confirmar que o PDF oficial de referência está disponível ou que há internet para
      abrir a publicação do INES.
- [ ] Fechar outros programas que possam estar usando a câmera do celular.

## Durante a demonstração

1. Explique que o protótipo reconhece configurações de mão cadastradas, não traduz
   conversas completas em Libras.
2. Mostre o alfabeto oficial ao lado do treinador e informe quais letras estão no
   modelo atual.
3. Faça uma previsão e aguarde a confirmação temporal antes de interpretar o sinal.
4. Use o modo desafio para mostrar acertos e erros da sessão.
5. Mostre o resumo JSON gerado, sem expor nomes, imagens ou vídeos de participantes.

## Se a câmera não abrir

- [ ] Não reiniciar várias instâncias do programa.
- [ ] Encerrar o aplicativo que está usando a câmera do celular.
- [ ] Tentar o índice alternativo, por exemplo `python treinar_visual.py --camera 1`.
- [ ] Continuar a apresentação com `--check`, os relatórios e o roteiro gravado.
