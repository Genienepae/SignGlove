import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.training.treinar_modelo import carregar_dataset


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
