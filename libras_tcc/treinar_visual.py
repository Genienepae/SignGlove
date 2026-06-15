"""
treinar_visual.py
=================
Interface gráfica completa para treinar a IA de Libras.

Como usar:
    python treinar_visual.py

Não precisa saber programar. Não precisa usar o terminal.
Tudo é feito por botões e cliques na tela.

Fluxo de uso:
    1. Clique em uma letra do alfabeto (ou digite no campo)
    2. Clique em "● GRAVAR" — a gravação inicia automaticamente após contagem
    3. Mantenha o sinal na câmera até a barra completar
    4. Repita para cada gesto que quiser ensinar
    5. Clique em "🧠 TREINAR IA" quando terminar
    6. Clique em "▶ TESTAR" para ver funcionando

Requisitos:
    pip install mediapipe opencv-python scikit-learn numpy
"""

import cv2
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
import json
import os
import pickle
import time
from collections import deque, Counter

# ── importa os módulos do projeto ──────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from core.detector import HandDetector
from core.features import extrair_features, dedos_levantados

# sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# ── PATHS ──────────────────────────────────────────────────────────────────
DIR_DADOS  = os.path.join(os.path.dirname(__file__), 'data', 'gestures')
DIR_MODEL  = os.path.join(os.path.dirname(__file__), 'models')
PATH_MODEL = os.path.join(DIR_MODEL, 'modelo_libras.pkl')
os.makedirs(DIR_DADOS, exist_ok=True)
os.makedirs(DIR_MODEL, exist_ok=True)

# ── CORES (tema escuro) ─────────────────────────────────────────────────────
BG      = '#0d1117'
SURFACE = '#161b22'
BORDER  = '#30363d'
GREEN   = '#00e676'
GREEN2  = '#00c853'
BLUE    = '#2979ff'
RED     = '#ff1744'
YELLOW  = '#ffea00'
TEXT    = '#e6edf3'
MUTED   = '#7d8590'
ORANGE  = '#ff9100'

# Alfabeto de Libras (letras mais comuns primeiro)
ALFABETO = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')


# ═══════════════════════════════════════════════════════════════════════════
class TreinadorLibras:
    """Aplicação principal — janela única com webcam + controles."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('🤟 Treinador de Libras — IA')
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(1100, 680)

        # Estado
        self.modo        = 'idle'       # idle | contagem | gravando | testando
        self.gesto_atual = ''
        self.amostras    = {}           # {nome: [features, ...]}
        self.modelo      = None
        self.le          = None
        self.treinado    = False
        self.cap         = None
        self.detector    = None
        self.rodando     = False
        self.meta_amostras = 80         # ← padrão menor = mais rápido

        # Contagem regressiva
        self._contagem_restante = 0

        # Buffer para teste
        self._buf_teste = deque(maxlen=10)

        self._carregar_dados_existentes()
        self._build_ui()
        self._iniciar_camera()

        self.root.protocol('WM_DELETE_WINDOW', self._fechar)
        self.root.mainloop()

    # ── DADOS ────────────────────────────────────────────────────────────
    def _carregar_dados_existentes(self):
        """Carrega arquivos .json já gravados anteriormente."""
        for arq in os.listdir(DIR_DADOS):
            if arq.endswith('.json'):
                nome = arq[:-5]
                with open(os.path.join(DIR_DADOS, arq)) as f:
                    dados = json.load(f)
                self.amostras[nome] = [np.array(d) for d in dados]
        if self.amostras:
            print(f'[OK] {len(self.amostras)} gestos carregados: {list(self.amostras.keys())}')

        # Tenta carregar modelo salvo
        if os.path.exists(PATH_MODEL):
            try:
                with open(PATH_MODEL, 'rb') as f:
                    dados = pickle.load(f)
                self.modelo = dados['modelo']
                self.le     = dados['le']
                self.treinado = True
                print(f'[OK] Modelo carregado: {list(self.le.classes_)}')
            except Exception:
                pass

    def _salvar_gesto(self, nome, lista_features):
        path = os.path.join(DIR_DADOS, f'{nome}.json')
        existente = []
        if os.path.exists(path):
            with open(path) as f:
                existente = json.load(f)
        existente += [f.tolist() for f in lista_features]
        with open(path, 'w') as f:
            json.dump(existente, f)
        self.amostras[nome] = [np.array(d) for d in existente]

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        # ── PAINEL ESQUERDO — webcam ──────────────────────────────────────
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # barra de status superior
        self.status_bar = tk.Label(left, text='AGUARDANDO', font=('Courier', 11, 'bold'),
                                   bg=SURFACE, fg=MUTED, pady=8)
        self.status_bar.grid(row=0, column=0, sticky='ew')

        # label que exibe o frame da webcam
        self.cam_label = tk.Label(left, bg='#000000', cursor='none')
        self.cam_label.grid(row=1, column=0, sticky='nsew')

        # barra de progresso de gravação
        prog_frame = tk.Frame(left, bg=BG, height=10)
        prog_frame.grid(row=2, column=0, sticky='ew')
        self.prog_bar = tk.Canvas(prog_frame, height=10, bg=BORDER,
                                  highlightthickness=0)
        self.prog_bar.pack(fill='x')
        self.prog_fill = self.prog_bar.create_rectangle(0, 0, 0, 10, fill=GREEN, outline='')

        # ── PAINEL DIREITO — controles ────────────────────────────────────
        right = tk.Frame(self.root, bg=SURFACE, width=340)
        right.grid(row=0, column=1, sticky='nsew')
        right.grid_propagate(False)
        right.grid_columnconfigure(0, weight=1)

        # scrollable canvas para caber todo o conteúdo
        canvas_right = tk.Canvas(right, bg=SURFACE, highlightthickness=0)
        scrollbar = tk.Scrollbar(right, orient='vertical', command=canvas_right.yview)
        self.scroll_frame = tk.Frame(canvas_right, bg=SURFACE)
        self.scroll_frame.bind('<Configure>',
            lambda e: canvas_right.configure(scrollregion=canvas_right.bbox('all')))
        canvas_right.create_window((0, 0), window=self.scroll_frame, anchor='nw')
        canvas_right.configure(yscrollcommand=scrollbar.set)
        canvas_right.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        sf = self.scroll_frame
        pad = {'padx': 16}

        # título
        tk.Label(sf, text='🤟 TREINADOR', font=('Courier', 14, 'bold'),
                 bg=SURFACE, fg=GREEN).pack(pady=(16, 2), **pad, anchor='w')
        tk.Label(sf, text='IA LIBRAS', font=('Courier', 9),
                 bg=SURFACE, fg=MUTED).pack(**pad, anchor='w')

        self._sep(sf)

        # ─ ATALHOS DO ALFABETO ────────────────────────────────────────────
        tk.Label(sf, text='ESCOLHA A LETRA', font=('Courier', 9, 'bold'),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', **pad, pady=(10, 6))

        alfa_outer = tk.Frame(sf, bg=SURFACE)
        alfa_outer.pack(fill='x', padx=16)

        self._btn_letra_refs = {}
        for i, letra in enumerate(ALFABETO):
            row_i = i // 7
            col_i = i % 7
            if col_i == 0:
                row_frame = tk.Frame(alfa_outer, bg=SURFACE)
                row_frame.pack(fill='x', pady=1)
            btn = tk.Button(row_frame, text=letra, width=3,
                            font=('Courier', 10, 'bold'),
                            bg=BG, fg=MUTED, relief='flat', cursor='hand2',
                            activebackground=GREEN2, activeforeground=BG,
                            command=lambda l=letra: self._selecionar_letra(l))
            btn.pack(side='left', padx=1)
            self._btn_letra_refs[letra] = btn

        self._sep(sf)

        # ─ GESTO PERSONALIZADO ───────────────────────────────────────────
        tk.Label(sf, text='OU NOME PERSONALIZADO', font=('Courier', 8, 'bold'),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', **pad, pady=(8, 4))

        entry_frame = tk.Frame(sf, bg=BORDER, padx=1, pady=1)
        entry_frame.pack(fill='x', padx=16)
        self.entry_gesto = tk.Entry(entry_frame, font=('Courier', 15, 'bold'),
                                    bg='#0d1117', fg=GREEN, insertbackground=GREEN,
                                    relief='flat', width=10)
        self.entry_gesto.pack(fill='x', ipady=6, padx=2, pady=2)
        self.entry_gesto.insert(0, 'A')

        # ─ META DE AMOSTRAS ──────────────────────────────────────────────
        self._sep(sf)
        tk.Label(sf, text='QUANTIDADE DE AMOSTRAS', font=('Courier', 9, 'bold'),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', **pad, pady=(8, 4))

        # Botões rápidos de quantidade
        qtd_frame = tk.Frame(sf, bg=SURFACE)
        qtd_frame.pack(fill='x', padx=16, pady=(0, 6))
        self._btns_qtd = {}
        for qtd, label in [(50, 'Rápido\n50'), (80, 'Normal\n80'), (150, 'Preciso\n150'), (300, 'Máximo\n300')]:
            b = tk.Button(qtd_frame, text=label, font=('Courier', 8, 'bold'),
                          bg=BG if qtd != 80 else GREEN, fg=MUTED if qtd != 80 else BG,
                          relief='flat', cursor='hand2', padx=4, pady=4,
                          command=lambda q=qtd: self._set_meta_rapida(q))
            b.pack(side='left', fill='x', expand=True, padx=1)
            self._btns_qtd[qtd] = b

        self.lbl_meta = tk.Label(sf, text='Meta: 80 amostras (~16s)',
                                 font=('Courier', 8), bg=SURFACE, fg=MUTED)
        self.lbl_meta.pack(anchor='w', padx=16)

        self._sep(sf)

        # ─ BOTÃO GRAVAR ──────────────────────────────────────────────────
        self.btn_gravar = self._botao(sf, '● GRAVAR GESTO', GREEN, BG,
                                      self._toggle_gravacao, size=13)

        # dica do espaço (agora opcional)
        tk.Label(sf, text='Gravação inicia automática! ESPAÇO pausa.',
                 font=('Courier', 8), bg=SURFACE, fg=MUTED).pack(padx=16, pady=(3, 0), anchor='w')

        # ─ GESTOS GRAVADOS ───────────────────────────────────────────────
        self._sep(sf)
        tk.Label(sf, text='GESTOS GRAVADOS', font=('Courier', 9, 'bold'),
                 bg=SURFACE, fg=MUTED).pack(anchor='w', **pad, pady=(8, 6))

        list_frame = tk.Frame(sf, bg=BG, relief='flat', bd=0)
        list_frame.pack(fill='x', padx=16)

        self.lista_gestos = tk.Frame(list_frame, bg=BG)
        self.lista_gestos.pack(fill='x')
        self._atualizar_lista_gestos()

        self._sep(sf)

        # ─ TREINAR ───────────────────────────────────────────────────────
        self.btn_treinar = self._botao(sf, '🧠 TREINAR IA', BLUE, TEXT,
                                       self._treinar, size=13)

        # barra de acurácia
        acc_frame = tk.Frame(sf, bg=SURFACE)
        acc_frame.pack(fill='x', padx=16, pady=(6, 0))
        tk.Label(acc_frame, text='Acurácia:', font=('Courier', 9),
                 bg=SURFACE, fg=MUTED).pack(side='left')
        self.lbl_acc = tk.Label(acc_frame, text='—', font=('Courier', 9, 'bold'),
                                bg=SURFACE, fg=GREEN)
        self.lbl_acc.pack(side='left', padx=8)

        self._sep(sf)

        # ─ TESTAR ────────────────────────────────────────────────────────
        self.btn_testar = self._botao(sf, '▶ TESTAR EM TEMPO REAL', ORANGE, BG,
                                      self._toggle_teste, size=12)

        # resultado do teste
        self.lbl_resultado = tk.Label(sf, text='—', font=('Courier', 22, 'bold'),
                                      bg=SURFACE, fg=GREEN)
        self.lbl_resultado.pack(pady=(10, 0))
        self.lbl_confianca = tk.Label(sf, text='', font=('Courier', 9),
                                      bg=SURFACE, fg=MUTED)
        self.lbl_confianca.pack()

        self._sep(sf)

        # ─ DICAS ─────────────────────────────────────────────────────────
        ajuda = (
            'DICAS:\n'
            '• Grave com boa iluminação\n'
            '• Mova a mão levemente\n'
            '• Repita de ângulos diferentes\n'
            '• Mínimo 2 gestos para treinar\n'
            '• "Normal" já é suficiente!'
        )
        tk.Label(sf, text=ajuda, font=('Courier', 8), bg=SURFACE,
                 fg=MUTED, justify='left').pack(padx=16, pady=(0, 20), anchor='w')

        # bind ESPAÇO na janela toda
        self.root.bind('<space>', lambda e: self._espaco_pressionado())

    def _sep(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill='x', padx=16, pady=8)

    def _botao(self, parent, texto, bg, fg, cmd, size=12):
        btn = tk.Button(parent, text=texto, font=('Courier', size, 'bold'),
                        bg=bg, fg=fg, relief='flat', cursor='hand2',
                        activebackground=GREEN2, activeforeground=BG,
                        padx=14, pady=10, command=cmd)
        btn.pack(fill='x', padx=16, pady=(4, 0))
        return btn

    def _selecionar_letra(self, letra):
        """Clique num botão do alfabeto: preenche o campo e destaca o botão."""
        # Reseta cor de todos
        for l, b in self._btn_letra_refs.items():
            n = len(self.amostras.get(l, []))
            cor = GREEN if n >= 80 else YELLOW if n >= 30 else BG
            b.config(bg=cor, fg=BG if n >= 80 else TEXT if n >= 30 else MUTED)

        # Destaca a selecionada
        self._btn_letra_refs[letra].config(bg=BLUE, fg=TEXT)
        self.entry_gesto.delete(0, tk.END)
        self.entry_gesto.insert(0, letra)

    def _set_meta_rapida(self, qtd):
        self.meta_amostras = qtd
        segs = max(1, qtd // 5)
        self.lbl_meta.config(text=f'Meta: {qtd} amostras (~{segs}s)')
        for q, b in self._btns_qtd.items():
            if q == qtd:
                b.config(bg=GREEN, fg=BG)
            else:
                b.config(bg=BG, fg=MUTED)

    def _atualizar_lista_gestos(self):
        for w in self.lista_gestos.winfo_children():
            w.destroy()

        if not self.amostras:
            tk.Label(self.lista_gestos, text='Nenhum gesto ainda',
                     font=('Courier', 9), bg=BG, fg=MUTED).pack(anchor='w')
            return

        for nome, feats in sorted(self.amostras.items()):
            n = len(feats)
            cor = GREEN if n >= 80 else YELLOW if n >= 30 else RED
            row = tk.Frame(self.lista_gestos, bg=BG)
            row.pack(fill='x', pady=2)

            tk.Label(row, text=f'  {nome}', font=('Courier', 11, 'bold'),
                     bg=BG, fg=TEXT, width=8, anchor='w').pack(side='left')

            # mini barra de progresso
            pct = min(n / 80, 1.0)
            bar_canvas = tk.Canvas(row, width=60, height=8, bg=BORDER,
                                   highlightthickness=0)
            bar_canvas.pack(side='left', padx=4)
            bar_canvas.create_rectangle(0, 0, int(60 * pct), 8, fill=cor, outline='')

            tk.Label(row, text=f'{n}', font=('Courier', 9),
                     bg=BG, fg=cor).pack(side='left', padx=2)

            # botão apagar
            tk.Button(row, text='✕', font=('Courier', 8), bg=BG, fg=MUTED,
                      relief='flat', cursor='hand2', padx=4,
                      command=lambda g=nome: self._apagar_gesto(g)).pack(side='right')

        # Atualiza cores dos botões do alfabeto
        for letra, btn in self._btn_letra_refs.items():
            n = len(self.amostras.get(letra, []))
            if n >= 80:
                btn.config(bg=GREEN, fg=BG)
            elif n >= 30:
                btn.config(bg=YELLOW, fg=BG)
            else:
                btn.config(bg=BG, fg=MUTED)

    def _apagar_gesto(self, nome):
        if not messagebox.askyesno('Apagar', f'Apagar todas as amostras de "{nome}"?'):
            return
        path = os.path.join(DIR_DADOS, f'{nome}.json')
        if os.path.exists(path):
            os.remove(path)
        self.amostras.pop(nome, None)
        self._atualizar_lista_gestos()

    # ── CÂMERA ───────────────────────────────────────────────────────────
    def _iniciar_camera(self):
        self.detector = HandDetector(min_detection_confidence=0.75)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  800)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
        self.rodando = True
        self._thread_cam = threading.Thread(target=self._loop_camera, daemon=True)
        self._thread_cam.start()

    def _loop_camera(self):
        """Thread de captura — roda em paralelo com a UI."""
        novas_amostras  = []
        coletando       = False
        t_inicio        = 0

        while self.rodando:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)
            landmarks, frame_ann, detectou = self.detector.detect(frame)
            h, w = frame_ann.shape[:2]

            # ── MODO CONTAGEM REGRESSIVA ──────────────────────────────────
            if self.modo == 'contagem':
                cv2.rectangle(frame_ann, (0, 0), (w, h), (13, 17, 23), -1)
                cnt = self._contagem_restante
                # número grande centralizado
                txt = str(cnt) if cnt > 0 else 'VÁ!'
                cor = (0, 230, 118) if cnt == 0 else (255, 234, 0)
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 5, 6)
                cv2.putText(frame_ann, txt,
                            ((w - tw) // 2, (h + th) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, cor, 6)
                sub = f'Prepare o gesto: {self.gesto_atual}'
                (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.putText(frame_ann, sub,
                            ((w - sw) // 2, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (125, 133, 144), 2)

                if cnt == 0:
                    # Transição para gravação
                    self.modo = 'gravando'
                    coletando = True
                    novas_amostras = []

            # ── MODO GRAVAÇÃO ─────────────────────────────────────────────
            elif self.modo == 'gravando':
                gesto = self.gesto_atual

                # HUD — nome do gesto
                cv2.rectangle(frame_ann, (0, 0), (w, 72), (13, 17, 23), -1)
                cv2.putText(frame_ann, f'GESTO: {gesto}', (12, 36),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 230, 118), 2)

                # progresso
                total_meta = self.meta_amostras
                n_atual = len(novas_amostras)
                pct = min(n_atual / total_meta, 1.0)

                if coletando:
                    status = f'● {n_atual}/{total_meta}'
                    cor_status = (0, 230, 118)
                else:
                    status = f'❚❚ PAUSADO — ESPAÇO para continuar'
                    cor_status = (255, 234, 0)

                cv2.putText(frame_ann, status, (12, 62),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, cor_status, 1)

                # barra de progresso no frame
                bw = int((w - 4) * pct)
                cv2.rectangle(frame_ann, (2, h - 10), (2 + bw, h), (0, 230, 118), -1)

                # dedos levantados
                if detectou and landmarks is not None:
                    dedos = dedos_levantados(landmarks)
                    nomes = ['P', 'I', 'M', 'A', 'Mi']
                    for i, (nm, lev) in enumerate(zip(nomes, dedos)):
                        cor = (0, 230, 118) if lev else (50, 50, 50)
                        cx, cy = 14 + i * 38, h - 26
                        cv2.circle(frame_ann, (cx, cy), 14, cor, -1)
                        cv2.putText(frame_ann, nm, (cx - 8, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

                    if coletando:
                        feat = extrair_features(landmarks)
                        if feat is not None:
                            novas_amostras.append(feat)

                        self.root.after(0, self._set_progresso, pct)

                        if len(novas_amostras) >= total_meta:
                            coletando = False
                            self._finalizar_gravacao(gesto, novas_amostras)
                            novas_amostras = []
                else:
                    cv2.putText(frame_ann, '⚠  MÃO NÃO DETECTADA', (12, h - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

                # ESPAÇO alterna coleta
                if self._espaco_flag:
                    self._espaco_flag = False
                    coletando = not coletando
                    if not coletando and novas_amostras:
                        # Parou manualmente
                        pass

            # ── MODO TESTE ────────────────────────────────────────────────
            elif self.modo == 'testando':
                cv2.rectangle(frame_ann, (0, 0), (w, 50), (13, 17, 23), -1)
                cv2.putText(frame_ann, 'MODO TESTE  —  Faça um sinal', (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 145, 0), 2)

                if detectou and landmarks is not None and self.treinado:
                    feat = extrair_features(landmarks)
                    if feat is not None:
                        probs = self.modelo.predict_proba(feat.reshape(1, -1))[0]
                        idx   = np.argmax(probs)
                        conf  = probs[idx]
                        pred  = self.le.inverse_transform([idx])[0]

                        self._buf_teste.append(pred if conf >= 0.70 else None)

                        validos = [x for x in self._buf_teste if x]
                        if validos:
                            mais, cnt = Counter(validos).most_common(1)[0]
                            if cnt / len(self._buf_teste) >= 0.6:
                                self.root.after(0, self._mostrar_resultado, mais, conf)

                        if conf >= 0.70:
                            cv2.putText(frame_ann, pred, (12, h - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 230, 118), 3)
                            cv2.putText(frame_ann, f'{conf*100:.0f}%', (12, h - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 118), 2)
                        else:
                            cv2.putText(frame_ann, '?', (12, h - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2.5, (100, 100, 100), 3)

            # ── MODO IDLE ─────────────────────────────────────────────────
            else:
                cv2.rectangle(frame_ann, (0, 0), (w, 40), (13, 17, 23), -1)
                msg = 'PRONTO' if self.treinado else 'Selecione uma letra e clique em GRAVAR'
                cv2.putText(frame_ann, msg, (12, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (125, 133, 144), 1)

            # exibe no tk
            frame_rgb = cv2.cvtColor(frame_ann, cv2.COLOR_BGR2RGB)
            img = self._frame_para_tk(frame_rgb)
            self.root.after(0, self._update_cam, img)

            time.sleep(0.01)

    def _frame_para_tk(self, frame_rgb):
        from PIL import Image, ImageTk
        img = Image.fromarray(frame_rgb)
        iw, ih = img.size
        lw = max(self.cam_label.winfo_width(), 400)
        lh = max(self.cam_label.winfo_height(), 300)
        ratio = min(lw / iw, lh / ih)
        img = img.resize((int(iw * ratio), int(ih * ratio)), Image.LANCZOS)
        return ImageTk.PhotoImage(img)

    def _update_cam(self, img):
        self.cam_label.configure(image=img)
        self.cam_label.image = img

    def _set_progresso(self, pct):
        w = self.prog_bar.winfo_width()
        self.prog_bar.coords(self.prog_fill, 0, 0, int(w * pct), 10)

    # ── GRAVAÇÃO COM CONTAGEM REGRESSIVA ─────────────────────────────────
    _espaco_flag = False

    def _espaco_pressionado(self):
        self._espaco_flag = True

    def _toggle_gravacao(self):
        nome = self.entry_gesto.get().strip().upper()
        if not nome:
            messagebox.showwarning('Atenção', 'Digite o nome do gesto primeiro.')
            return
        if self.modo == 'testando':
            self._toggle_teste()

        if self.modo in ('gravando', 'contagem'):
            # Para
            self.modo = 'idle'
            self.btn_gravar.config(text='● GRAVAR GESTO', bg=GREEN)
            self.status_bar.config(text='GRAVAÇÃO CANCELADA', fg=MUTED)
            self._set_progresso(0)
        else:
            # Inicia contagem regressiva
            self.gesto_atual = nome
            self.modo = 'contagem'
            self._espaco_flag = False
            self.btn_gravar.config(text='■ CANCELAR', bg=RED)
            self.status_bar.config(
                text=f'Prepare-se para gravar "{nome}"...', fg=YELLOW)
            self._iniciar_contagem()

    def _iniciar_contagem(self):
        """Dispara a contagem regressiva 3-2-1-VÁ! na UI e na câmera."""
        self._contagem_restante = 3
        self._tick_contagem()

    def _tick_contagem(self):
        if self.modo != 'contagem':
            return
        cnt = self._contagem_restante
        if cnt > 0:
            self.status_bar.config(
                text=f'Preparando "{self.gesto_atual}"... {cnt}', fg=YELLOW)
            self._contagem_restante -= 1
            self.root.after(1000, self._tick_contagem)
        else:
            # cnt == 0: mostra "VÁ!" por 800ms na câmera, depois a thread assume
            self._contagem_restante = 0
            self.status_bar.config(
                text=f'● GRAVANDO "{self.gesto_atual}"', fg=GREEN)
            # A thread de câmera detecta contagem_restante==0 e muda para 'gravando'

    def _finalizar_gravacao(self, nome, amostras):
        """Salva as amostras e atualiza a UI (pode ser chamado da thread)."""
        self._salvar_gesto(nome, amostras)
        self.root.after(0, self._pos_gravacao, nome, len(amostras))

    def _pos_gravacao(self, nome, n):
        self.modo = 'idle'
        self.btn_gravar.config(text='● GRAVAR GESTO', bg=GREEN)
        self._set_progresso(0)
        self._atualizar_lista_gestos()
        self.status_bar.config(
            text=f'✓  {n} amostras de "{nome}" salvas!  Grave outro gesto ou treine a IA.', fg=GREEN)

    # ── TREINO ───────────────────────────────────────────────────────────
    def _treinar(self):
        if len(self.amostras) < 2:
            messagebox.showwarning('Atenção',
                'Grave pelo menos 2 gestos diferentes antes de treinar.')
            return
        for nome, feats in self.amostras.items():
            if len(feats) < 20:
                messagebox.showwarning('Atenção',
                    f'O gesto "{nome}" tem só {len(feats)} amostras.\n'
                    'Recomendado: pelo menos 30 amostras por gesto.')
                return

        self.btn_treinar.config(text='⏳ Treinando...', state='disabled')
        self.status_bar.config(text='Treinando a IA... aguarde.', fg=YELLOW)
        threading.Thread(target=self._treinar_thread, daemon=True).start()

    def _treinar_thread(self):
        try:
            X, y = [], []
            for nome, feats in self.amostras.items():
                for f in feats:
                    X.append(f)
                    y.append(nome)

            X = np.array(X, dtype=np.float32)
            y = np.array(y)

            # ── Aumentação simples (espelhar + ruído) ──
            X_aug, y_aug = [X.copy()], [y.copy()]
            for _ in range(2):
                Xn = X.copy()
                Xn[:, :63] += np.random.normal(0, 0.015, Xn[:, :63].shape).astype(np.float32)
                pts = Xn[:, :63].reshape(-1, 21, 3)
                pts[:, :, 0] = -pts[:, :, 0]
                Xn[:, :63] = pts.reshape(-1, 63)
                X_aug.append(Xn)
                y_aug.append(y.copy())

            X_all = np.vstack(X_aug)
            y_all = np.concatenate(y_aug)

            le = LabelEncoder()
            y_enc = le.fit_transform(y_all)

            modelo = Pipeline([
                ('sc', StandardScaler()),
                ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                             probability=True, random_state=42))
            ])

            scores = cross_val_score(modelo, X_all, y_enc, cv=3, scoring='accuracy')
            acc = scores.mean()

            modelo.fit(X_all, y_enc)

            os.makedirs(DIR_MODEL, exist_ok=True)
            with open(PATH_MODEL, 'wb') as f:
                pickle.dump({'modelo': modelo, 'le': le}, f)

            self.modelo   = modelo
            self.le       = le
            self.treinado = True

            self.root.after(0, self._pos_treino, acc, len(le.classes_))

        except Exception as e:
            self.root.after(0, messagebox.showerror, 'Erro no treino', str(e))
            self.root.after(0, self.btn_treinar.config,
                            {'text': '🧠 TREINAR IA', 'state': 'normal'})

    def _pos_treino(self, acc, n_classes):
        self.btn_treinar.config(text='🧠 TREINAR IA', state='normal')
        self.lbl_acc.config(text=f'{acc*100:.1f}%',
                             fg=GREEN if acc >= 0.85 else YELLOW if acc >= 0.70 else RED)
        self.status_bar.config(
            text=f'✓  Treinamento concluído!  {n_classes} gestos · Acurácia: {acc*100:.1f}%',
            fg=GREEN)
        messagebox.showinfo('Treinamento concluído!',
            f'✅ A IA foi treinada com sucesso!\n\n'
            f'   Gestos: {list(self.le.classes_)}\n'
            f'   Acurácia estimada: {acc*100:.1f}%\n\n'
            f'Clique em "▶ TESTAR" para ver funcionando.')

    # ── TESTE ─────────────────────────────────────────────────────────────
    def _toggle_teste(self):
        if not self.treinado:
            messagebox.showwarning('Atenção', 'Treine a IA antes de testar.')
            return
        if self.modo == 'testando':
            self.modo = 'idle'
            self.btn_testar.config(text='▶ TESTAR EM TEMPO REAL')
            self.lbl_resultado.config(text='—')
            self.lbl_confianca.config(text='')
            self.status_bar.config(text='PRONTO', fg=MUTED)
        else:
            if self.modo in ('gravando', 'contagem'):
                self._toggle_gravacao()
            self.modo = 'testando'
            self._buf_teste.clear()
            self.btn_testar.config(text='■ PARAR TESTE')
            self.status_bar.config(text='TESTANDO EM TEMPO REAL — faça um sinal para a câmera', fg=ORANGE)

    def _mostrar_resultado(self, gesto, conf):
        self.lbl_resultado.config(text=gesto)
        cor = GREEN if conf >= 0.85 else YELLOW if conf >= 0.70 else RED
        self.lbl_resultado.config(fg=cor)
        self.lbl_confianca.config(text=f'Confiança: {conf*100:.0f}%', fg=MUTED)

    # ── ENCERRAR ──────────────────────────────────────────────────────────
    def _fechar(self):
        self.rodando = False
        time.sleep(0.15)
        if self.cap:
            self.cap.release()
        if self.detector:
            self.detector.release()
        self.root.destroy()


# ══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print('\n❌  Pillow não encontrado.')
        print('   Instale com:  pip install Pillow')
        print('   Depois execute:  python treinar_visual.py\n')
        sys.exit(1)

    TreinadorLibras()
