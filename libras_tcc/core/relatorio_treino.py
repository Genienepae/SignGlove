"""Formata o resumo JSON do treinamento para leitura no terminal."""

from __future__ import annotations


def formatar_resumo_treino(resumo: dict) -> str:
    obrigatorios = {'acuracia', 'f1_macro', 'criterio_metrica', 'gestos', 'amostras',
                    'por_gesto', 'matriz_confusao'}
    faltantes = obrigatorios.difference(resumo)
    if faltantes:
        raise ValueError(f'Resumo de treino sem campos: {", ".join(sorted(faltantes))}.')

    linhas = [
        'RESUMO DO TREINAMENTO',
        f"Amostras: {int(resumo['amostras'])}",
        f"Critério: {resumo['criterio_metrica']}",
        f"Acurácia: {float(resumo['acuracia']) * 100:.1f}%",
        f"F1 macro: {float(resumo['f1_macro']) * 100:.1f}%",
        '',
        'POR GESTO',
        'Gesto | Precisão | Recall | F1 | Amostras',
    ]
    for gesto in resumo['gestos']:
        metricas = resumo['por_gesto'][gesto]
        linhas.append(
            f"{gesto:5} | {float(metricas['precisao']) * 100:8.1f}% | "
            f"{float(metricas['recall']) * 100:6.1f}% | "
            f"{float(metricas['f1']) * 100:5.1f}% | {int(metricas['amostras']):8}"
        )

    matriz = resumo['matriz_confusao']
    rotulos = matriz['rotulos']
    valores = matriz['valores']
    if len(valores) != len(rotulos) or any(len(linha) != len(rotulos) for linha in valores):
        raise ValueError('Matriz de confusão incompatível com os rótulos.')
    linhas.extend(['', 'MATRIZ DE CONFUSÃO (linhas: real; colunas: previsão)',
                   '      ' + ' '.join(f'{rotulo:>5}' for rotulo in rotulos)])
    for rotulo, linha in zip(rotulos, valores):
        linhas.append(f'{rotulo:>5} ' + ' '.join(f'{int(valor):5}' for valor in linha))
    return '\n'.join(linhas)
