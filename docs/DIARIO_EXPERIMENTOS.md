# Diário de experimentos

Use uma cópia desta ficha para cada rodada de coleta, treino ou prática. O
diário registra o procedimento e as métricas sem identificar os participantes.

## Identificação

| Campo | Registro |
|---|---|
| Experimento | `EXP-____` |
| Data e horário | ____ |
| Responsável | ____ |
| Versão/commit do código | ____ |
| Modelo usado | SVM / KNN / outro: ____ |
| Gestos avaliados | ____ |

## Coleta

| Campo | Registro |
|---|---|
| Códigos anônimos | P__ / P__ / P__ |
| Sessões | S__ / S__ |
| Amostras por gesto | ____ |
| Iluminação e distância | ____ |
| Observações de qualidade | ____ |

## Resultado

| Métrica | Valor |
|---|---:|
| Acurácia | ____ % |
| F1 macro | ____ % |
| Taxa de acerto na prática | ____ % |
| Duração da prática | ____ s |
| Gesto mais confundido | ____ |

Comandos úteis para preencher a ficha:

```powershell
python training/diagnosticar_projeto.py --strict
python training/avaliar_por_participante.py --saida reports/EXP-____.json
python training/resumo_praticas.py --saida reports/praticas-EXP-____.json
```

Não registre nomes, imagens ou vídeos no diário. Para participantes humanos,
combine o procedimento e o consentimento com o orientador e use os mesmos
códigos anônimos no manifesto de coleta e nesta ficha.
