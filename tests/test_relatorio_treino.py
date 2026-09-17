import unittest

from libras_tcc.core.relatorio_treino import formatar_resumo_treino


class RelatorioTreinoTests(unittest.TestCase):
    def resumo(self):
        return {
            'acuracia': 0.8,
            'f1_macro': 0.75,
            'criterio_metrica': 'por amostra',
            'gestos': ['A', 'B'],
            'amostras': 10,
            'por_gesto': {
                'A': {'precisao': 0.8, 'recall': 0.9, 'f1': 0.85, 'amostras': 5},
                'B': {'precisao': 0.7, 'recall': 0.6, 'f1': 0.65, 'amostras': 5},
            },
            'matriz_confusao': {'rotulos': ['A', 'B'], 'valores': [[4, 1], [2, 3]]},
        }

    def test_formata_metricas_e_matriz(self):
        texto = formatar_resumo_treino(self.resumo())
        self.assertIn('Acurácia: 80.0%', texto)
        self.assertIn('F1 macro: 75.0%', texto)
        self.assertIn('MATRIZ DE CONFUSÃO', texto)

    def test_rejeita_matriz_com_dimensao_invalida(self):
        resumo = self.resumo()
        resumo['matriz_confusao']['valores'] = [[1]]
        with self.assertRaisesRegex(ValueError, 'Matriz de confusão'):
            formatar_resumo_treino(resumo)
