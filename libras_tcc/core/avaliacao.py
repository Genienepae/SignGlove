"""Avaliação que separa participantes inteiros entre treino e teste."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def carregar_dataset_por_participante(pasta_amostras: str | Path, manifesto: str | Path):
    """Monta X, y e grupos apenas dos lotes identificados no manifesto."""
    pasta_amostras = Path(pasta_amostras)
    manifesto = Path(manifesto)
    if not manifesto.exists():
        raise ValueError('Nenhum manifesto de coleta foi encontrado.')

    arquivos = {}
    features, classes, grupos = [], [], []
    for numero_linha, linha in enumerate(manifesto.read_text(encoding='utf-8').splitlines(), start=1):
        if not linha.strip():
            continue
        try:
            lote = json.loads(linha)
            nome_arquivo = lote['arquivo_amostras']
            inicio = int(lote['indice_inicio'])
            fim = int(lote['indice_fim'])
            gesto = lote['gesto']
            participante = lote['participante']
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as erro:
            raise ValueError(f'Metadado inválido na linha {numero_linha}: {erro}') from erro

        if inicio < 0 or fim < inicio:
            raise ValueError(f'Intervalo inválido na linha {numero_linha}.')
        if nome_arquivo not in arquivos:
            caminho = pasta_amostras / nome_arquivo
            if not caminho.exists():
                raise ValueError(f'Amostras não encontradas: {caminho}')
            arquivos[nome_arquivo] = json.loads(caminho.read_text(encoding='utf-8'))
        amostras = arquivos[nome_arquivo]
        if fim >= len(amostras):
            raise ValueError(f'Intervalo fora do arquivo na linha {numero_linha}.')

        for amostra in amostras[inicio:fim + 1]:
            features.append(amostra)
            classes.append(gesto)
            grupos.append(participante)

    if not features:
        raise ValueError('O manifesto não possui lotes de amostras.')
    return np.asarray(features, dtype=float), np.asarray(classes), np.asarray(grupos)


def divisao_por_participante_disponivel(classes, grupos) -> bool:
    """Confirma que cada treino mantém todas as classes necessárias."""
    participantes = np.unique(grupos)
    classes = np.asarray(classes)
    if len(participantes) < 2 or len(np.unique(classes)) < 2:
        return False
    todas_classes = set(classes)
    divisao = GroupKFold(n_splits=min(5, len(participantes)))
    for treino, _ in divisao.split(np.zeros(len(classes)), classes, grupos):
        if set(classes[treino]) != todas_classes:
            return False
    return True


def avaliar_svm_por_participante(features, classes, grupos) -> dict:
    """Mede uma SVM em divisões que nunca misturam a mesma pessoa."""
    features = np.asarray(features, dtype=float)
    classes = np.asarray(classes)
    grupos = np.asarray(grupos)
    participantes = np.unique(grupos)
    if not divisao_por_participante_disponivel(classes, grupos):
        raise ValueError(
            'Colete todos os gestos com pelo menos 2 participantes para avaliar por pessoa.')

    divisao = GroupKFold(n_splits=min(5, len(participantes)))
    previsoes = np.empty(len(classes), dtype=classes.dtype)
    for treino, teste in divisao.split(features, classes, grupos):
        modelo = Pipeline([
            ('escala', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale')),
        ])
        modelo.fit(features[treino], classes[treino])
        previsoes[teste] = modelo.predict(features[teste])

    rotulos = list(np.unique(classes))
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        classes, previsoes, labels=rotulos, zero_division=0)
    por_gesto = {
        gesto: {
            'precisao': float(precisao[indice]),
            'recall': float(recall[indice]),
            'f1': float(f1[indice]),
            'amostras': int(suporte[indice]),
        }
        for indice, gesto in enumerate(rotulos)
    }
    return {
        'participantes': list(participantes),
        'amostras': int(len(classes)),
        'gestos': rotulos,
        'acuracia': float(accuracy_score(classes, previsoes)),
        'f1_macro': float(f1_score(classes, previsoes, average='macro', zero_division=0)),
        'por_gesto': por_gesto,
        'matriz_confusao': confusion_matrix(classes, previsoes, labels=rotulos).tolist(),
    }
