import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.core.diagnostico import diagnosticar_projeto


class DiagnosticoProjetoTests(unittest.TestCase):
    def test_resume_dados_modelo_e_cobertura(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            dados = raiz / 'gestures'
            dados.mkdir()
            (dados / 'A.json').write_text(json.dumps([[1]] * 20), encoding='utf-8')
            (dados / 'B.json').write_text(json.dumps([[2]] * 20), encoding='utf-8')
            modelo = raiz / 'modelo.pkl'
            modelo.write_bytes(b'modelo')
            manifesto = raiz / 'coletas.jsonl'
            manifesto.write_text('', encoding='utf-8')

            resultado = diagnosticar_projeto(dados, manifesto, modelo)
            self.assertEqual(resultado['total_amostras'], 40)
            self.assertTrue(resultado['modelo_encontrado'])
            self.assertTrue(resultado['pronto_treino'])
            self.assertFalse(resultado['avaliacao_por_participante_pronta'])
            self.assertEqual(resultado['dimensoes_features'], [1])
            self.assertEqual(resultado['amostras_invalidas'], 0)

    def test_marca_amostras_invalidas_e_nao_libera_treino(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            dados = raiz / 'gestures'
            dados.mkdir()
            (dados / 'A.json').write_text(json.dumps([[1, 2], [float('nan')]]), encoding='utf-8')
            (dados / 'B.json').write_text(json.dumps([[1, 2]] * 20), encoding='utf-8')
            manifesto = raiz / 'coletas.jsonl'
            manifesto.write_text('', encoding='utf-8')

            resultado = diagnosticar_projeto(dados, manifesto, raiz / 'modelo.pkl')
            self.assertEqual(resultado['amostras_invalidas'], 1)
            self.assertEqual(resultado['dimensoes_features'], [1, 2])
            self.assertFalse(resultado['pronto_treino'])
