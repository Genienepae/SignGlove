import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from libras_tcc.core.praticas import registrar_pratica


class RegistroPraticaTests(unittest.TestCase):
    def test_registra_sessao_anonima(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'praticas.jsonl'
            inicio = datetime.now(timezone.utc).isoformat()
            registro = registrar_pratica(arquivo, 'p01', inicio, 4, erros=2)
            salvo = json.loads(arquivo.read_text(encoding='utf-8'))

            self.assertEqual(registro['participante'], 'P01')
            self.assertEqual(salvo['acertos'], 4)
            self.assertEqual(salvo['erros'], 2)
            self.assertGreaterEqual(salvo['duracao_segundos'], 0)

    def test_rejeita_acertos_negativos(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaises(ValueError):
                registrar_pratica(Path(pasta) / 'x.jsonl', 'P01',
                                  datetime.now(timezone.utc).isoformat(), -1)
            with self.assertRaises(ValueError):
                registrar_pratica(Path(pasta) / 'x.jsonl', 'P01',
                                  datetime.now(timezone.utc).isoformat(), 0, erros=-1)
