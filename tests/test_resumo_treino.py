import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.training.treinar_modelo import salvar_resumo_treino


class ResumoTreinoTests(unittest.TestCase):
    def test_salva_metricas_principais(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'reports' / 'treino.json'
            por_gesto = {'A': {'precisao': 1.0, 'recall': 0.5, 'f1': 0.67, 'amostras': 2}}
            matriz = {'rotulos': ['A', 'B'], 'valores': [[1, 1], [0, 1]]}
            salvar_resumo_treino(
                arquivo, 0.875, 0.80, 'por amostra', ['B', 'A', 'A'], 3, por_gesto, matriz)
            resultado = json.loads(arquivo.read_text(encoding='utf-8'))
            self.assertEqual(resultado, {
                'acuracia': 0.875,
                'f1_macro': 0.80,
                'criterio_metrica': 'por amostra',
                'gestos': ['A', 'B'],
                'amostras': 3,
                'por_gesto': por_gesto,
                'matriz_confusao': matriz,
            })
