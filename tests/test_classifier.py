"""Regressões da confirmação temporal, sem webcam ou modelo serializado."""

import unittest
import pickle
import tempfile
from pathlib import Path

import numpy as np

from libras_tcc.core.classifier import ClassificadorGestos


class ModeloComProbabilidades:
    """Controla a saída do modelo para exercitar transições reais do buffer."""

    def __init__(self, probabilidades):
        self.probabilidades = iter(probabilidades)

    def predict_proba(self, features):
        return np.array([next(self.probabilidades)])


class ModeloFixo:
    def predict_proba(self, features):
        return np.array([[0.9, 0.1]])


class ConfirmacaoTemporalTests(unittest.TestCase):
    def criar_classificador(self, previsoes):
        probabilidades = {
            'A': [0.95, 0.05],
            'B': [0.05, 0.95],
            None: [0.50, 0.50],
        }
        clf = ClassificadorGestos(buffer_frames=8, confianca_minima=0.75)
        clf.label_encoder.fit(['A', 'B'])
        clf.modelo = ModeloComProbabilidades(probabilidades[p] for p in previsoes)
        clf.treinado = True
        return clf

    def prever(self, clf):
        return clf.prever(np.zeros(73, dtype=np.float32))

    def test_aguarda_buffer_completo(self):
        clf = self.criar_classificador(['A'] * 8)
        for _ in range(7):
            self.assertFalse(self.prever(clf)[2])
        self.assertEqual(self.prever(clf), ('A', 0.95, True))

    def test_troca_nao_herda_confirmacao_do_gesto_anterior(self):
        clf = self.criar_classificador(['A'] * 8 + ['B'] * 6)
        for _ in range(8):
            self.prever(clf)

        # A ainda domina o histórico nos primeiros frames de B.
        # Confirmar B aqui adicionaria uma letra instável ao texto.
        for _ in range(5):
            self.assertEqual(self.prever(clf), ('B', 0.95, False))
        self.assertEqual(self.prever(clf), ('B', 0.95, True))

    def test_oscilacao_entre_classes_nao_confirma(self):
        clf = self.criar_classificador(['A', 'B'] * 4)
        for _ in range(8):
            self.assertFalse(self.prever(clf)[2])

    def test_duvidas_contam_no_tamanho_total_da_janela(self):
        clf = self.criar_classificador([None] * 3 + ['A'] * 5)
        for _ in range(8):
            self.assertFalse(self.prever(clf)[2])

    def test_seis_acertos_em_oito_frames_confirmam(self):
        clf = self.criar_classificador([None] * 2 + ['A'] * 6)
        for _ in range(7):
            self.assertFalse(self.prever(clf)[2])
        self.assertEqual(self.prever(clf), ('A', 0.95, True))

    def test_baixa_confianca_nao_herda_confirmacao(self):
        clf = self.criar_classificador(['A'] * 8 + [None])
        for _ in range(8):
            self.prever(clf)
        self.assertEqual(self.prever(clf), (None, 0.5, False))

    def test_reset_exige_nova_estabilizacao(self):
        clf = self.criar_classificador(['A'] * 9)
        for _ in range(8):
            self.prever(clf)
        clf.resetar_buffer()
        self.assertFalse(self.prever(clf)[2])

    def test_carrega_formato_do_treinador_visual(self):
        original = self.criar_classificador(['A'])
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / 'modelo.pkl'
            with caminho.open('wb') as arquivo:
                pickle.dump({'modelo': ModeloFixo(), 'le': original.label_encoder,
                             'nome_modelo': 'SVM'}, arquivo)
            classificador = ClassificadorGestos(confianca_minima=0.8)
            classificador.carregar(str(caminho))

        self.assertEqual(list(classificador.label_encoder.classes_), ['A', 'B'])
        self.assertEqual(classificador.algoritmo, 'SVM')
        self.assertEqual(classificador.confianca_minima, 0.8)

    def test_carregar_limpa_buffer_do_modelo_anterior(self):
        original = self.criar_classificador(['A'] * 8)
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / 'modelo.pkl'
            with caminho.open('wb') as arquivo:
                pickle.dump({'modelo': ModeloFixo(), 'le': original.label_encoder}, arquivo)
            classificador = self.criar_classificador(['A'] * 8)
            for _ in range(8):
                self.prever(classificador)
            self.assertEqual(len(classificador._buffer), 8)
            classificador.carregar(str(caminho))
            self.assertEqual(len(classificador._buffer), 0)

    def test_salvar_registra_formato_e_dimensao(self):
        classificador = ClassificadorGestos()
        classificador.label_encoder.fit(['A', 'B'])
        classificador.modelo = ModeloFixo()
        classificador.treinado = True
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / 'modelo.pkl'
            classificador.salvar(str(caminho))
            with caminho.open('rb') as arquivo:
                salvo = pickle.load(arquivo)
        self.assertEqual(salvo['formato_modelo'], 2)
        self.assertIsNone(salvo['features_esperadas'])


if __name__ == '__main__':
    unittest.main()
