import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.core.relatorio_praticas import resumir_praticas


class RelatorioPraticasTests(unittest.TestCase):
    def test_agrega_sessoes_por_participante(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'praticas.jsonl'
            registros = [
                {'participante': 'P01', 'acertos': 3, 'erros': 1, 'duracao_segundos': 20},
                {'participante': 'P01', 'acertos': 2, 'erros': 2, 'duracao_segundos': 30},
                {'participante': 'P02', 'acertos': 4, 'erros': 0, 'duracao_segundos': 15},
            ]
            arquivo.write_text(''.join(json.dumps(item) + '\n' for item in registros), encoding='utf-8')

            resumo = resumir_praticas(arquivo)
            self.assertEqual(resumo['sessoes'], 3)
            self.assertEqual(resumo['participantes']['P01'], {
                'sessoes': 2, 'acertos': 5, 'erros': 3, 'duracao_segundos': 50,
            })

    def test_sem_arquivo_retorna_resumo_vazio(self):
        with tempfile.TemporaryDirectory() as pasta:
            self.assertEqual(resumir_praticas(Path(pasta) / 'ausente.jsonl'),
                             {'sessoes': 0, 'participantes': {}})
