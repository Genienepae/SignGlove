import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.core.classifier import ClassificadorGestos
from libras_tcc.training.treinar_modelo import avaliar_modelo, carregar_dataset


class CarregamentoTreinoTests(unittest.TestCase):
    def test_carrega_73_features(self):
        with tempfile.TemporaryDirectory() as pasta:
            dados = Path(pasta)
            (dados / 'A.json').write_text(json.dumps([[0.0] * 73] * 20), encoding='utf-8')
            matriz, classes = carregar_dataset(str(dados))
            self.assertEqual(matriz.shape, (20, 73))
            self.assertEqual(len(classes), 20)

    def test_rejeita_dimensao_incompativel(self):
        with tempfile.TemporaryDirectory() as pasta:
            dados = Path(pasta)
            (dados / 'A.json').write_text(json.dumps([[0.0] * 72] * 20), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '73 features'):
                carregar_dataset(str(dados))

    def test_avaliacao_por_participante_padroniza_matriz(self):
        with tempfile.TemporaryDirectory() as pasta:
            dados = Path(pasta) / 'gestures'
            dados.mkdir()
            manifesto = Path(pasta) / 'coletas.jsonl'
            lotes = []
            for gesto, base in [('A', 0.0), ('B', 1.0)]:
                (dados / f'{gesto}.json').write_text(
                    json.dumps([[base, base], [base + 0.1, base],
                                [base, base + 0.1], [base + 0.1, base + 0.1]]),
                    encoding='utf-8')
                lotes.extend([
                    {'participante': 'P01', 'gesto': gesto,
                     'arquivo_amostras': f'{gesto}.json', 'indice_inicio': 0, 'indice_fim': 1},
                    {'participante': 'P02', 'gesto': gesto,
                     'arquivo_amostras': f'{gesto}.json', 'indice_inicio': 2, 'indice_fim': 3},
                ])
            manifesto.write_text(''.join(json.dumps(lote) + '\n' for lote in lotes), encoding='utf-8')

            resultado = avaliar_modelo(
                ClassificadorGestos(), [], [], str(dados), str(manifesto))
            self.assertEqual(resultado['criterio_metrica'], 'por participante')
            self.assertEqual(resultado['matriz_confusao']['rotulos'], ['A', 'B'])
            self.assertEqual(len(resultado['matriz_confusao']['valores']), 2)
