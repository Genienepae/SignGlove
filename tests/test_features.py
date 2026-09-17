import unittest

import numpy as np

from libras_tcc.core.features import extrair_features


class ExtracaoFeaturesTests(unittest.TestCase):
    def test_rejeita_entrada_com_tamanho_incorreto(self):
        self.assertIsNone(extrair_features(np.zeros(62)))

    def test_rejeita_valor_nao_finito(self):
        landmarks = np.zeros(63)
        landmarks[0] = np.nan
        self.assertIsNone(extrair_features(landmarks))

    def test_gera_vetor_de_73_features(self):
        landmarks = np.arange(63, dtype=np.float32)
        resultado = extrair_features(landmarks)
        self.assertEqual(resultado.shape, (73,))
        self.assertTrue(np.isfinite(resultado).all())
