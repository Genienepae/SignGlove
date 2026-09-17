import json
import tempfile
import unittest
from pathlib import Path

from libras_tcc.core.avaliacao import (
    avaliar_svm_por_participante,
    carregar_dataset_por_participante,
    divisao_por_participante_disponivel,
    salvar_resultado_avaliacao,
)


class AvaliacaoPorParticipanteTests(unittest.TestCase):
    def test_carrega_apenas_intervalos_identificados(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz / 'A.json').write_text(json.dumps([[1, 2], [3, 4], [5, 6]]), encoding='utf-8')
            manifesto = raiz / 'coletas.jsonl'
            manifesto.write_text(
                json.dumps({
                    'participante': 'P01', 'gesto': 'A', 'arquivo_amostras': 'A.json',
                    'indice_inicio': 1, 'indice_fim': 2,
                }) + '\n', encoding='utf-8')

            x, y, grupos = carregar_dataset_por_participante(raiz, manifesto)
            self.assertEqual(x.tolist(), [[3.0, 4.0], [5.0, 6.0]])
            self.assertEqual(y.tolist(), ['A', 'A'])
            self.assertEqual(grupos.tolist(), ['P01', 'P01'])

    def test_rejeita_intervalo_fora_do_arquivo(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz / 'A.json').write_text(json.dumps([[1, 2]]), encoding='utf-8')
            manifesto = raiz / 'coletas.jsonl'
            manifesto.write_text(json.dumps({
                'participante': 'P01', 'gesto': 'A', 'arquivo_amostras': 'A.json',
                'indice_inicio': 0, 'indice_fim': 2,
            }) + '\n', encoding='utf-8')

            with self.assertRaises(ValueError):
                carregar_dataset_por_participante(raiz, manifesto)

    def test_exige_todos_os_gestos_em_cada_treino(self):
        self.assertTrue(divisao_por_participante_disponivel(
            ['A', 'B', 'A', 'B'], ['P01', 'P01', 'P02', 'P02']))
        self.assertFalse(divisao_por_participante_disponivel(
            ['A', 'B', 'A'], ['P01', 'P01', 'P02']))

    def test_avaliacao_retorna_metricas_por_gesto(self):
        x = [[0, 0], [0.1, 0], [1, 1], [1.1, 1], [0, 0.1], [0.1, 0.1], [1, 1.1], [1.1, 1.1]]
        y = ['A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
        grupos = ['P01', 'P01', 'P01', 'P01', 'P02', 'P02', 'P02', 'P02']
        resultado = avaliar_svm_por_participante(x, y, grupos)
        self.assertEqual(set(resultado['por_gesto']), {'A', 'B'})
        self.assertEqual(len(resultado['matriz_confusao']), 2)

    def test_salva_resultado_em_json(self):
        with tempfile.TemporaryDirectory() as pasta:
            arquivo = Path(pasta) / 'relatorios' / 'avaliacao.json'
            salvar_resultado_avaliacao(arquivo, {'acuracia': 0.75})
            self.assertEqual(json.loads(arquivo.read_text(encoding='utf-8')), {'acuracia': 0.75})
