import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.core.cobertura_coletas import participantes_completos, resumir_cobertura


class CoberturaColetasTests(unittest.TestCase):
    def test_soma_lotes_e_detecta_cobertura(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'coletas.jsonl'
            lotes = [
                {'participante': 'P01', 'gesto': 'A', 'quantidade': 20},
                {'participante': 'P01', 'gesto': 'B', 'quantidade': 10},
                {'participante': 'P02', 'gesto': 'A', 'quantidade': 30},
                {'participante': 'P02', 'gesto': 'B', 'quantidade': 25},
            ]
            arquivo.write_text(''.join(json.dumps(lote) + '\n' for lote in lotes), encoding='utf-8')
            resumo = resumir_cobertura(arquivo)
            self.assertEqual(resumo['participantes']['P01']['A'], 20)
            self.assertTrue(participantes_completos(resumo))

    def test_detecta_gesto_faltando(self):
        resumo = {'gestos': ['A', 'B'], 'participantes': {'P01': {'A': 20}, 'P02': {'A': 20, 'B': 20}}}
        self.assertFalse(participantes_completos(resumo))
