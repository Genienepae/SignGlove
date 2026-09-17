import tempfile
import unittest
from pathlib import Path

from libras_tcc.core.coletas import (
    normalizar_codigo_participante,
    registrar_lote,
    remover_registros_do_gesto,
)


class MetadadosColetaTests(unittest.TestCase):
    def test_normaliza_codigo_anonimo(self):
        self.assertEqual(normalizar_codigo_participante(' p-01 '), 'P-01')

    def test_rejeita_nome_com_espaco(self):
        with self.assertRaises(ValueError):
            normalizar_codigo_participante('Maria Silva')

    def test_registra_intervalo_e_remove_por_gesto(self):
        with tempfile.TemporaryDirectory() as pasta:
            manifesto = Path(pasta) / 'coletas.jsonl'
            registro = registrar_lote(manifesto, 'P01', 'A', 'A.json', 10, 3)
            registrar_lote(manifesto, 'P02', 'B', 'B.json', 0, 2)
            self.assertEqual(registro['indice_fim'], 12)

            remover_registros_do_gesto(manifesto, 'A.json')
            texto = manifesto.read_text(encoding='utf-8')
            self.assertNotIn('"gesto": "A"', texto)
            self.assertIn('"gesto": "B"', texto)
