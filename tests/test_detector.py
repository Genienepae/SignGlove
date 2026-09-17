import unittest

from libras_tcc.core.detector import CORES_DEDOS, HandDetector


class _Classe:
    def __init__(self, lado):
        self.label = lado


class _Classificacao:
    def __init__(self, lado):
        self.classification = [_Classe(lado)]


class _Resultado:
    def __init__(self, lados):
        self.multi_handedness = [_Classificacao(lado) for lado in lados]


class DetectorTests(unittest.TestCase):
    def test_preserva_lados_detectados_para_numerar_os_dedos(self):
        resultado = _Resultado(['Right', 'Left'])
        self.assertEqual(HandDetector._extrair_lados(resultado, 2), ['Right', 'Left'])

    def test_usa_direita_como_padrao_quando_mediapipe_nao_informa_lado(self):
        resultado = _Resultado([])
        self.assertEqual(HandDetector._extrair_lados(resultado, 2), ['Right', 'Right'])

    def test_existe_uma_cor_para_cada_dedo_numerado(self):
        self.assertEqual(len(CORES_DEDOS), 10)
        self.assertEqual(len(set(CORES_DEDOS)), 10)

