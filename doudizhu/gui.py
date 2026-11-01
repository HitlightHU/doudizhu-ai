r"""Tkinter GUI：人机对弈 + 分析模式（叫地主/加倍/出牌建议）。

启动：
  python -m doudizhu.gui
或双击 run_gui.bat
"""
from __future__ import annotations

import copy
import os
import queue
import random
import threading
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import torch

from .actions import N_RANKS, TYPE_NAMES, counts_to_mask, mask_to_counts
from .analysis import (AnalysisSession, bid_advice, bid_rollout, should_grab,
                       should_grab_noshuffle)
from .bidnet import load_bid_net, score_hands
from .engine import HIST, HIST_DIM, BatchEngine
from .model import load_checkpoint
from .reference import Game, action_str, canonicalize_mask, get_action_space, mask_str
from .selfplay import select_actions_batch
from .train import make_action_features

ROOT = Path(__file__).resolve().parent.parent
RANK_NAMES = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "小王", "大王"]
FONT = ("Microsoft YaHei UI", 12)
FONT_B = ("Microsoft YaHei UI", 12, "bold")
FONT_T = ("Microsoft YaHei UI", 16, "bold")
FONT_SMALL = ("Microsoft YaHei UI", 10)
FONT_CARD = ("Microsoft YaHei UI", 18, "bold")
FONT_HAND = ("Microsoft YaHei UI", 16, "bold")
FONT_TABLE = ("Consolas", 11)
FONT_LOG = ("Consolas", 10)
TOTAL_COUNTS = np.array([4] * 13 + [1, 1], dtype=np.int8)


def played_str(row) -> str:
    parts = []
    for r in range(N_RANKS):
        for _ in range(int(row[r])):
            parts.append(RANK_NAMES[r])
    return " ".join(parts) if parts else "—"


def action_desc(asp, a: int) -> str:
    if a == asp.n:
        return "过"
    return f"{TYPE_NAMES[int(asp.typ[a])]} {mask_str(int(asp.mask[a]))}"


def clone_game(g: Game) -> Game:
    """轻量快照：共享 ActionSpace，只复制可变状态。"""
    ng = copy.copy(g)
    ng.hands = list(g.hands)
    ng.played = g.played.copy()
    ng.history = list(g.history)
    return ng


def clone_session(s: AnalysisSession) -> AnalysisSession:
    ns = copy.copy(s)
    ns.played = s.played.copy()
    ns.history = list(s.history)
    ns.hand_sizes = list(s.hand_sizes)
    return ns


class CardPicker(ttk.Frame):
    """按 rank 加牌的输入控件：大号点数 + 下方剩余张数。

    limits[r]：该点数最多还能选几张；max_total：总张数上限。
    选到上限后按钮自动禁用；超过上限的点击/程序化设置都会被夹紧。
    """

    def __init__(self, master, on_change=None, cols=8, show_total=True,
                 max_total=None, max_counts=None, sort_desc=False, rank_desc=False):
        super().__init__(master)
        self.sort_desc = bool(sort_desc)
        self.rank_desc = bool(rank_desc)
        self.counts = np.zeros(N_RANKS, dtype=np.int8)
        self.stack = []
        self.on_change = on_change
        self.cols = cols
        self.show_total = show_total
        self.limits = np.array(max_counts if max_counts is not None else TOTAL_COUNTS,
                               dtype=np.int8)
        self.max_total = None if max_total is None else int(max_total)
        self.buttons = [None] * N_RANKS
        self.rem_labels = [None] * N_RANKS
        self.hand_strip = ttk.Frame(self, height=100, width=520)
        self.hand_strip.grid(row=0, column=0, columnspan=cols, sticky="we", pady=(0, 4))
        self.hand_strip.grid_propagate(False)
        self.chip_buttons = []
        ttk.Label(self.hand_strip, text="手牌：", font=FONT_SMALL).grid(
            row=0, column=0, padx=(0, 2), sticky="nw")
        self.empty_lbl = ttk.Label(self.hand_strip, text="（未选）", font=FONT_SMALL,
                                   foreground="#888")
        self.empty_lbl.grid(row=0, column=1, sticky="nw")
        order = range(N_RANKS - 1, -1, -1) if self.rank_desc else range(N_RANKS)
        for pos, r in enumerate(order):
            cell = ttk.Frame(self)
            cell.grid(row=1 + pos // cols, column=pos % cols, padx=1, pady=1)
            b = tk.Button(cell, text=RANK_NAMES[r], font=FONT_CARD, width=3,
                          height=1, relief="raised", bd=1,
                          command=lambda rr=r: self.add(rr))
            b.bind("<Button-3>", lambda e, rr=r: self.remove_rank(rr))
            b.pack()
            lab = ttk.Label(cell, text="剩 0", font=FONT_SMALL, foreground="#666")
            lab.pack()
            self.buttons[r] = b
            self.rem_labels[r] = lab
        self.info = ttk.Label(self, text="", font=FONT_B, wraplength=540, justify="left")
        self.info.grid(row=3, column=0, columnspan=cols, sticky="w", pady=(6, 0))
        btns = ttk.Frame(self)
        btns.grid(row=4, column=0, columnspan=cols, sticky="w")
        ttk.Button(btns, text="撤销", command=self.undo).pack(side="left", padx=2)
        ttk.Button(btns, text="清空", command=self.clear).pack(side="left", padx=2)
        self.refresh()

    def _make_chip(self, r):
        return tk.Button(self.hand_strip, text=RANK_NAMES[r], font=FONT_HAND,
                         width=2 if r < 12 else 3, height=1, relief="groove", bd=1,
                         bg="#fff3cd", activebackground="#ffe08a",
                         command=lambda rr=r: self.remove_rank(rr))

    def _layout_chips(self):
        for i, (_, b) in enumerate(self.chip_buttons):
            b.grid(row=i // 10, column=1 + i % 10, padx=1, pady=1)
        if self.chip_buttons:
            self.empty_lbl.grid_remove()
        else:
            self.empty_lbl.grid()

    def _rebuild_chips(self):
        for _, b in self.chip_buttons:
            b.destroy()
        self.chip_buttons = []
        cards = []
        for r in range(N_RANKS):
            cards.extend([r] * int(self.counts[r]))
        if self.sort_desc:
            cards.reverse()
        for r in cards:
            self.chip_buttons.append((r, self._make_chip(r)))
        self._layout_chips()

    def _add_chip(self, r):
        self.chip_buttons.append((r, self._make_chip(r)))
        self.chip_buttons.sort(key=lambda t: t[0], reverse=self.sort_desc)
        self._layout_chips()

    def _remove_chip(self, r):
        for i in range(len(self.chip_buttons) - 1, -1, -1):
            if self.chip_buttons[i][0] == r:
                _, b = self.chip_buttons.pop(i)
                b.destroy()
                break
        self._layout_chips()

    def _refresh_controls(self):
        total = int(self.counts.sum())
        cap = (self.max_total is not None and total >= self.max_total)
        for r, b in enumerate(self.buttons):
            rem = int(self.limits[r] - self.counts[r])
            self.rem_labels[r].configure(
                text=f"剩 {max(0, rem)}",
                foreground=("#c62828" if rem <= 0 else "#666"))
            b.configure(state=("disabled" if (rem <= 0 or cap) else "normal"))
        if self.show_total:
            suffix = f" / {self.max_total}" if self.max_total is not None else ""
            self.info.configure(text=f"已选 {total}{suffix} 张（点手牌栏的大牌可放回一张）")
        else:
            self.info.configure(text=f"已选 {total} 张")
        if self.on_change:
            self.on_change()

    def set_limits(self, max_counts, max_total):
        """设置每个点数的可选上限和总张数上限，并夹紧当前已选。"""
        lim = np.asarray(max_counts, dtype=np.int16)
        lim = np.clip(lim, 0, 4)
        lim[13] = min(int(lim[13]), 1)
        lim[14] = min(int(lim[14]), 1)
        self.limits = lim.astype(np.int8)
        self.max_total = None if max_total is None else max(0, int(max_total))
        self.counts = np.minimum(self.counts, self.limits)
        while self.max_total is not None and int(self.counts.sum()) > self.max_total:
            r = int(np.argmax(self.counts))
            if self.counts[r] <= 0:
                break
            self.counts[r] -= 1
        self.stack = []
        self._rebuild_chips()
        self._refresh_controls()

    def add(self, r: int):
        if int(self.limits[r]) <= int(self.counts[r]):
            return
        if self.max_total is not None and self.total() >= self.max_total:
            return
        self.counts[r] += 1
        self.stack.append(r)
        self._add_chip(r)
        self._refresh_controls()

    def remove_rank(self, r: int):
        """点已选的牌放回：从该点数减一张（右键点上方点数按钮同效）。"""
        if self.counts[r] <= 0:
            return
        self.counts[r] -= 1
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i] == r:
                del self.stack[i]
                break
        self._remove_chip(r)
        self._refresh_controls()

    def remove_at(self, idx: int):
        cards = []
        for r in range(N_RANKS):
            cards.extend([r] * int(self.counts[r]))
        if 0 <= idx < len(cards):
            self.remove_rank(cards[idx])

    def undo(self):
        if self.stack:
            r = self.stack.pop()
            if self.counts[r] > 0:
                self.counts[r] -= 1
                self._remove_chip(r)
            self._refresh_controls()

    def clear(self):
        self.counts[:] = 0
        self.stack.clear()
        self._rebuild_chips()
        self._refresh_controls()

    def set_counts(self, counts):
        c = np.asarray(counts, dtype=np.int16).copy()
        c = np.clip(c, 0, self.limits)
        while self.max_total is not None and int(c.sum()) > self.max_total:
            r = int(np.argmax(c))
            if c[r] <= 0:
                break
            c[r] -= 1
        self.counts = c.astype(np.int8)
        self.stack = []
        self._rebuild_chips()
        self._refresh_controls()

    def refresh(self):
        self._rebuild_chips()
        self._refresh_controls()

    def text(self) -> str:
        parts = []
        for r in range(N_RANKS):
            for _ in range(int(self.counts[r])):
                parts.append(RANK_NAMES[r])
        return " ".join(parts)

    def mask(self) -> int:
        return counts_to_mask(self.counts)

    def total(self) -> int:
        return int(self.counts.sum())


class ScrollFrame(ttk.Frame):
    def __init__(self, master, height=220):
        super().__init__(master)
        self.canvas = tk.Canvas(self, height=height, highlightthickness=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll.pack(side="right", fill="y")

    def clear(self):
        for w in self.inner.winfo_children():
            w.destroy()


class HumanFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.asp = app.asp
        self.game: Game | None = None
        self.phase = "idle"
        self.human_seat = 0
        self.hand17 = 0
        self.rec_active = False
        self._last_turn = -1
        self.history = []
        self.hcursor = -1
        self.ended = False
        self._after_id = None
        self._live_after_id = None
        self._live_token = 0
        self._search_gen = 0
        self._search_event = None
        self._search_queue = queue.Queue()
        self._search_poll = None
        self.engine = BatchEngine(1, seed=1)

        top = ttk.Frame(self)
        top.pack(fill="x")
        self.status = ttk.Label(top, text="点击「新开一局」开始", font=FONT_T)
        self.status.pack(side="left")
        ttk.Button(top, text="新开一局", command=self.new_game).pack(side="right", padx=2)

        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, pady=4)
        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(body, width=570)
        right.pack(side="right", fill="both", padx=(8, 0))

        self.table = tk.Text(left, height=24, width=56, font=FONT_TABLE, state="disabled")
        self.table.pack(fill="both", expand=True)
        self.log = tk.Text(left, height=8, font=FONT_LOG, state="disabled")
        self.log.pack(fill="x", pady=(4, 0))

        self.hand_lbl = ttk.Label(right, text="手牌：—", font=FONT_B, wraplength=550)
        self.hand_lbl.pack(fill="x")
        self.bid_box = ttk.LabelFrame(right, text="叫地主", padding=6)
        self.bid_box.pack(fill="x", pady=4)
        self.bid_lbl = ttk.Label(self.bid_box, text="—", font=FONT_B)
        self.bid_lbl.pack(fill="x")

        self.act_box = ttk.LabelFrame(right, text="出牌 / 推荐", padding=4)
        self.act_box.pack(fill="x", expand=False, pady=4)
        self.actions = ScrollFrame(self.act_box, height=120)
        self.actions.pack(fill="both", expand=True)

        self.play_box = ttk.Frame(right)
        self.picker = CardPicker(self.play_box, max_total=20)
        self.picker.pack(fill="x", pady=4)
        row = ttk.Frame(self.play_box)
        row.pack(fill="x")
        self.btn_play = ttk.Button(row, text="出牌", command=self.manual_play)
        self.btn_play.pack(side="left", padx=2)
        self.btn_pass = ttk.Button(row, text="过", command=self.manual_pass)
        self.btn_pass.pack(side="left", padx=2)
        self.btn_rec = ttk.Button(row, text="推荐前三", command=self.show_recommend)
        self.btn_rec.pack(side="left", padx=2)
        self.btn_wr = ttk.Button(row, text="更新胜率", command=self.update_winrate)
        self.btn_wr.pack(side="left", padx=2)
        self.win_lbl = ttk.Label(self.play_box, text="胜率：—", font=FONT_B)
        self.win_lbl.pack(fill="x", pady=2)
        self.camp_lbl = ttk.Label(self.play_box, text="实时估计：地主 —  |  农民 —", font=FONT_B)
        self.camp_lbl.pack(fill="x", pady=2)
        self.depth = tk.StringVar(value="64")
        ttk.Label(self.play_box, text="推理深度（rollouts）").pack(anchor="w")
        ttk.Combobox(self.play_box, textvariable=self.depth, values=["8", "16", "32", "64", "128"],
                     state="readonly", width=8).pack(anchor="w")
        self.turn_badge = ttk.Label(self.play_box, text="", font=("Microsoft YaHei UI", 20, "bold"),
                                    foreground="#c62828")
        self.turn_badge.pack(anchor="e", pady=(4, 0))
        self._set_actions([])
        self.picker.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
        self._search_poll = self.after(100, self._poll_search)

    # ---------- 工具 ----------
    def logmsg(self, s):
        self.log.configure(state="normal")
        self.log.insert("end", s + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def set_status(self, s):
        self.status.configure(text=s)

    def _set_actions(self, items):
        self.actions.clear()
        if not items:
            ttk.Label(self.actions.inner, text="（暂无）").pack(anchor="w")
            return
        for text, cb in items:
            ttk.Button(self.actions.inner, text=text, command=cb).pack(fill="x", pady=1)

    def show_table(self):
        g = self.game
        if g is None:
            txt = ""
        else:
            lines = []
            for p in range(3):
                role = "未定" if g.landlord < 0 else ("地主" if p == g.landlord else "农民")
                mark = " <-- 当前" if p == g.turn and g.winner < 0 and not self.ended else ""
                you = "（你）" if p == self.human_seat else ""
                lines.append(f"玩家{p}{you} {role} 剩{g.card_count(p):2d}张{mark}")
            lines.append("")
            if self.phase == "bid":
                lines.append("底牌：未公开（叫地主后公布）")
            else:
                lines.append(f"底牌：{mask_str(g.bottom)}")
            lines.append(f"上一手：{action_desc(self.asp, g.last_act) if g.last_act >= 0 else '无（自由出牌）'}")
            lines.append("")
            for p in range(3):
                lines.append(f"玩家{p} 已出：{played_str(g.played[p])}")
            if g.winner >= 0:
                lines.append("")
                lines.append(f"*** 玩家{g.winner} 获胜 ***")
            if self.ended and g.winner < 0:
                lines.append("")
                lines.append("*** 牌局已手动结束 ***")
            txt = "\n".join(lines)
        self.table.configure(state="normal")
        self.table.delete("1.0", "end")
        self.table.insert("1.0", txt)
        self.table.configure(state="disabled")

    def refresh(self):
        self.show_table()
        if self.game is None:
            self.picker.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
            self._sync_turn_badge()
            return
        self.hand_lbl.configure(text=f"你的手牌（{self.game.card_count(self.human_seat)}）："
                                     f"{mask_str(self.game.hands[self.human_seat])}")
        if self.ended:
            self.picker.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
            self._sync_turn_badge()
            return
        # 只允许选自己手里的牌；轮到自己时开放，其他时候禁用
        turn = int(self.game.turn) if self.game.winner < 0 else -1
        if turn != self._last_turn:
            self.picker.clear()
            self._last_turn = turn
        if self.phase == "play" and self.game.winner < 0 and turn == self.human_seat:
            lim = mask_to_counts(self.game.hands[self.human_seat])
            self.picker.set_limits(lim, max(1, int(lim.sum())))
        else:
            self.picker.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
        self._sync_buttons()
        self._sync_turn_badge()

    def _sync_turn_badge(self):
        g = self.game
        if g is None:
            text, font, color = "", FONT, "#555"
        elif self.ended or g.winner >= 0:
            text, font, color = "牌局结束", ("Microsoft YaHei UI", 14, "bold"), "#555"
        elif self.phase == "play" and g.turn == self.human_seat:
            text, font, color = "该你出牌！！！", ("Microsoft YaHei UI", 24, "bold"), "#c62828"
        elif self.phase == "play":
            rel = (int(g.turn) - self.human_seat) % 3
            side = "下家" if rel == 1 else "上家"
            text, font, color = f"该你的{side}出牌", ("Microsoft YaHei UI", 20, "bold"), "#444"
        else:
            text, font, color = "", FONT, "#555"
        self.turn_badge.configure(text=text, font=font, foreground=color)

    def _sync_buttons(self):
        g = self.game
        active = (g is not None and self.phase == "play" and not self.ended
                  and g.winner < 0 and g.turn == self.human_seat)
        for b in (self.btn_play, self.btn_rec, self.btn_wr):
            b.configure(state=("normal" if active else "disabled"))
        pass_ok = active and (self.asp.n in g.legal_actions())
        self.btn_pass.configure(state=("normal" if pass_ok else "disabled"))

    # ---------- 历史 / 撤销 ----------
    def _cancel_pending(self):
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _cancel_live(self):
        if self._live_after_id is not None:
            try:
                self.after_cancel(self._live_after_id)
            except Exception:
                pass
            self._live_after_id = None
        self._live_token += 1

    def _schedule_live(self, ms=80):
        self._cancel_live()
        token = self._live_token
        self._live_after_id = self.after(ms, lambda: self._run_live(token))

    def _run_live(self, token):
        self._live_after_id = None
        if token != self._live_token:
            return
        if self.game is None or self.ended or self.phase != "play":
            return
        try:
            s = self.make_session()
            ll, ff = s.camp_value(self.app.model, self.app.device, self.app.action_feat,
                                  samples=32)
        except Exception:
            return
        if token != self._live_token:
            return
        self._set_camp(ll, ff)

    def _set_camp(self, ll, ff):
        g = self.game
        if g is None:
            return
        if g.winner >= 0:
            self.camp_lbl.configure(
                text=f"牌局结束：{'地主' if g.winner == g.landlord else '农民'}胜")
            return
        you = "地主" if g.landlord == self.human_seat else "农民"
        self.camp_lbl.configure(
            text=f"实时估计：地主 {ll:.1%}  |  农民 {ff:.1%}  （你是{you}）")

    def _cancel_search(self):
        self._search_gen += 1
        ev = self._search_event
        self._search_event = None
        if ev is not None:
            ev.set()
        self._cancel_live()

    def _start_search(self):
        self._cancel_search()
        if self.game is None or self.ended or self.phase != "play" or self.game.winner >= 0:
            return
        sess = self.make_session()
        if sess.turn == sess.human_seat:
            self._start_top3_search(sess)
        else:
            self._start_camp_search(sess)

    def _start_top3_search(self, sess):
        ev = threading.Event()
        self._search_event = ev
        gen = self._search_gen
        q = self._search_queue
        model = self.app.model
        device = self.app.device
        feat = self.app.action_feat
        depth = int(self.depth.get())

        def run():
            try:
                def cb(best, pol, samples, elapsed):
                    q.put(("top3", gen, best, pol, samples, elapsed))
                sess.candidates(model, device, feat, topk=3,
                                samples=depth, max_samples=512, time_budget_ms=6000,
                                progress_cb=cb, cancel_event=ev, publish_ms=100)
            except Exception as e:
                q.put(("error", gen, str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _start_camp_search(self, sess):
        ev = threading.Event()
        self._search_event = ev
        gen = self._search_gen
        q = self._search_queue
        model = self.app.model
        device = self.app.device
        feat = self.app.action_feat

        def run():
            n = 8
            try:
                while not ev.is_set() and n <= 512:
                    ll, ff = sess.camp_value(model, device, feat, samples=n)
                    q.put(("camp", gen, ll, ff, n))
                    n = min(512, n * 2)
                    if ev.wait(0.2):
                        break
            except Exception as e:
                q.put(("error", gen, str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _poll_search(self):
        if not self.winfo_exists():
            return
        while True:
            try:
                item = self._search_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            gen = item[1]
            if gen != self._search_gen:
                continue
            if kind == "top3" and self.game is not None and not self.ended:
                _, _, best, pol, samples, elapsed = item
                items = []
                for a, v in best:
                    items.append((f"{action_desc(self.asp, a)}    胜率 {v:.1%}",
                                  lambda aa=a: self.play_action(aa)))
                self._set_actions(items)
                self.rec_active = True
                self.win_lbl.configure(
                    text=f"实时前三选（{samples} 次采样 / {elapsed:.0f} ms）最高 {pol:.1%}")
                if self.game.landlord == self.human_seat:
                    self._set_camp(pol, 1.0 - pol)
                else:
                    self._set_camp(1.0 - pol, pol)
                self.set_status("实时前三选：每 0.2s 更新；也可手动出任意合法牌")
            elif kind == "camp" and self.game is not None and not self.ended:
                _, _, ll, ff, samples = item
                self._set_camp(ll, ff)
            elif kind == "bid" and self.phase == "bid":
                _, _, p_ll, p_f, samples = item
                rec = "建议抢地主" if should_grab(p_ll, p_f) else "建议不抢"
                self.bid_lbl.configure(
                    text=f"地主胜率 {p_ll:.1%}    农民胜率 {p_f:.1%}"
                         f"（实时 {samples} 次采样）\n"
                         f"规则：地主胜率>农民 / 地主>70% / 低于农民不超5% → {rec}")
            elif kind == "error":
                self.set_status(f"实时搜索失败：{item[2]}")
        self._search_poll = self.after(100, self._poll_search)

    def on_close(self):
        self._cancel_pending()
        self._cancel_search()
        if self._search_poll is not None:
            try:
                self.after_cancel(self._search_poll)
            except Exception:
                pass
            self._search_poll = None

    def _schedule(self, ms):
        self._cancel_pending()
        self._after_id = self.after(ms, self.ai_step)

    def _record(self):
        if self.game is None:
            return
        self.history = self.history[:self.hcursor + 1]
        self.history.append(clone_game(self.game))
        self.hcursor = len(self.history) - 1

    def _restore(self, idx):
        if not (0 <= idx < len(self.history)):
            return
        self._cancel_pending()
        self._cancel_search()
        self.game = clone_game(self.history[idx])
        self.hcursor = idx
        self.phase = "play"
        self.rec_active = False
        self.ended = False
        self.picker.clear()
        self._last_turn = -1
        self.camp_lbl.configure(text="实时估计：计算中…")
        self.refresh()
        self._start_search()
        g = self.game
        if g.winner >= 0:
            self._set_actions([("再来一局", self.new_game)])
            self.set_status(f"玩家{g.winner} 获胜")
            return
        if g.turn == self.human_seat:
            self.ai_step()
        elif idx == len(self.history) - 1:
            self.set_status("已回退，AI 即将行动")
            self._schedule(300)
        else:
            self.set_status("已回退到 AI 回合，点「下一回合」继续")

    def prev_turn(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束，请新开一局")
            return
        if not self.history:
            messagebox.showinfo("提示", "牌局还没开始")
            return
        if self.hcursor <= 0:
            messagebox.showinfo("提示", "已经是牌局开始")
            return
        self.picker.clear()
        self._restore(self.hcursor - 1)

    def next_turn(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束，请新开一局")
            return
        if not self.history:
            messagebox.showinfo("提示", "牌局还没开始")
            return
        if self.hcursor >= len(self.history) - 1:
            messagebox.showinfo("提示", "没有下一回合")
            return
        self._restore(self.hcursor + 1)

    def undo(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束，请新开一局")
            return
        if self.picker.total() > 0:
            self.picker.clear()
            self.logmsg("撤销：清空本次选牌")
            self.set_status("已清空选牌")
            return
        self.prev_turn()

    def end_game(self):
        if self.game is None:
            messagebox.showinfo("提示", "还没有牌局")
            return
        self._cancel_pending()
        self._cancel_search()
        self.ended = True
        self.phase = "done"
        self.rec_active = False
        self.picker.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
        self._set_actions([("再来一局", self.new_game)])
        self.set_status("牌局已手动结束")
        if self.game.winner >= 0:
            ll = 1.0 if self.game.winner == self.game.landlord else 0.0
            self._set_camp(ll, 1.0 - ll)
        else:
            self.camp_lbl.configure(text="实时估计：—（已结束）")
        self.refresh()

    def ribbon(self, action):
        if action == "reset":
            self.new_game()
        elif action == "end":
            self.end_game()
        elif action == "undo":
            self.undo()
        elif action == "prev":
            self.prev_turn()
        elif action == "next":
            self.next_turn()

    # ---------- 新局与叫地主 ----------
    def new_game(self):
        self._cancel_pending()
        self._cancel_search()
        self.camp_lbl.configure(text="实时估计：地主 —  |  农民 —")
        self.history = []
        self.hcursor = -1
        self.ended = False
        self.game = Game.deal(seed=random.randrange(1 << 30))
        self.phase = "bid"
        self.rec_active = False
        self._last_turn = -1
        self.act_box.configure(text="叫地主 / 加倍")
        self.hand17 = self.game.hands[self.human_seat]
        self.play_box.pack_forget()
        self.bid_box.pack_forget()
        self.bid_box.pack(fill="x", pady=4, before=self.act_box)
        self.bid_lbl.configure(text="计算叫地主建议…")
        self.refresh()
        self._set_actions([])
        self.logmsg("—— 新的一局，你是玩家0，17 张手牌 ——")
        p_ll, p_f, grab = bid_advice(self.app.bid_net, self.hand17, self.app.device)
        rec = "建议抢地主" if grab else "建议不抢"
        self.bid_lbl.configure(
            text=f"地主胜率 {p_ll:.1%}    农民胜率 {p_f:.1%}\n"
                 f"规则：地主胜率>农民 / 地主>70% / 低于农民不超5% → {rec}")
        self._set_actions([
            ("抢地主", lambda: self.finish_bid(True)),
            ("不抢", lambda: self.finish_bid(False)),
        ])
        self.set_status("叫地主：请选择")
        self._start_bid_search()

    def _start_bid_search(self):
        self._cancel_search()
        ev = threading.Event()
        self._search_event = ev
        gen = self._search_gen
        q = self._search_queue
        hand = self.hand17
        model = self.app.model
        device = self.app.device
        feat = self.app.action_feat
        asp = self.asp

        def run():
            n = 8
            try:
                while not ev.is_set() and n <= 512:
                    p_ll, p_f = bid_rollout(hand, model, device, feat,
                                            samples=n, seed=random.randrange(1 << 30), asp=asp)
                    q.put(("bid", gen, p_ll, p_f, n))
                    n = min(512, n * 2)
                    if ev.wait(0.2):
                        break
            except Exception as e:
                q.put(("error", gen, str(e)))

        threading.Thread(target=run, daemon=True).start()

    def finish_bid(self, grab: bool):
        self._cancel_search()
        if self.phase != "bid":
            return
        g = self.game
        if grab:
            landlord = self.human_seat
        else:
            ai = [p for p in range(3) if p != self.human_seat]
            masks = np.array([g.hands[p] for p in ai], dtype=np.int64)
            scores = score_hands(self.app.bid_net, masks, self.app.device) if self.app.bid_net else np.full((2, 2), 0.5)
            landlord = ai[int(np.argmax(scores[:, 0]))]
        g.assign_landlord(landlord)
        self.phase = "double"
        self.logmsg(f"地主是玩家{landlord}（{'你' if landlord == self.human_seat else 'AI'}），底牌 {mask_str(g.bottom)}")
        self.refresh()
        self.bid_lbl.configure(text="计算加倍建议…")
        self.set_status("加倍阶段")
        sess = self.make_session()
        wr = sess.policy_value(self.app.model, self.app.device, self.app.action_feat, samples=48)
        self.double_wr = wr
        rec = "建议加倍" if wr > 0.70 else "不建议加倍"
        self.bid_lbl.configure(
            text=f"底牌 {mask_str(g.bottom)}\n你的初始胜率估计 {wr:.1%} → {rec}\n"
                 f"（胜率 > 70% 时推荐加倍）")
        self._set_actions([
            ("加倍", lambda: self.start_play("加倍")),
            ("不加倍", lambda: self.start_play("不加倍")),
        ])

    def start_play(self, choice):
        self.phase = "play"
        self.ended = False
        self.act_box.configure(text="出牌 / 推荐")
        self.play_box.pack(fill="x", pady=4)
        self.history = []
        self.hcursor = -1
        self._record()
        self.logmsg(f"你选择：{choice}")
        self.bid_box.pack_forget()
        self.set_status("出牌阶段")
        self.refresh()
        self.camp_lbl.configure(text="实时估计：计算中…")
        self._start_search()
        self._schedule(100)

    # ---------- 出牌 ----------
    def make_session(self) -> AnalysisSession:
        g = self.game
        s = AnalysisSession()
        s.human_seat = self.human_seat
        s.landlord = g.landlord
        s.bottom = g.bottom
        s.human_hand = g.hands[self.human_seat]
        s.hand_sizes = [g.card_count(p) for p in range(3)]
        s.played = np.array(g.played, dtype=np.int8)
        s.turn = g.turn
        s.last_act = g.last_act
        s.last_player = g.last_player
        s.passes = g.passes
        s.history = [(p, (-1 if a == self.asp.n else int(a))) for p, a in g.history]
        s.winner = g.winner
        return s

    def show_recommend(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束")
            return
        if self.game is None or self.phase != "play" or self.game.turn != self.human_seat:
            messagebox.showinfo("提示", "现在不是你的出牌回合")
            return
        self.set_status("实时前三选搜索中…（每 0.2s 更新）")
        self._start_search()

    def update_winrate(self):
        if self.ended or self.game is None or self.game.winner >= 0:
            return
        self.win_lbl.configure(text="实时胜率计算中…")
        self.set_status("实时搜索中…（每 0.2s 更新）")
        self._start_search()

    def model_action(self) -> int:
        g = self.game
        e = self.engine
        e.hands[0, 0] = g.hands[0]; e.hands[0, 1] = g.hands[1]; e.hands[0, 2] = g.hands[2]
        e.played[0] = g.played
        e.bottom[0] = g.bottom
        e.landlord[0] = g.landlord
        e.turn[0] = g.turn
        e.last_act[0] = g.last_act
        e.last_player[0] = g.last_player
        e.passes[0] = g.passes
        e.winner[0] = g.winner
        hist = g.history[-HIST:]
        e.hist_len[0] = len(hist)
        for i in range(HIST):
            e.hist_a[0, i] = -1
            e.hist_p[0, i] = -1
        for i, (p, a) in enumerate(hist):
            e.hist_a[0, i] = -1 if a == self.asp.n else int(a)
            e.hist_p[0, i] = int(p)
        st, hh = e.observe()
        idx, cnt = e.legal()
        a = int(select_actions_batch(self.app.model, self.app.device, self.app.action_feat,
                                     st, hh, idx, cnt, 0.0)[0])
        legal = g.legal_actions()
        if a not in legal:
            a = legal[0]
        return a

    def ai_step(self):
        self._after_id = None
        if self.game is None or self.phase != "play" or self.ended:
            return
        g = self.game
        if g.winner >= 0:
            self._cancel_search()
            self.refresh()
            self._set_actions([("再来一局", self.new_game)])
            ll = 1.0 if g.winner == g.landlord else 0.0
            self._set_camp(ll, 1.0 - ll)
            self.set_status(f"玩家{g.winner} 获胜")
            return
        if g.turn == self.human_seat:
            self.refresh()
            if self.rec_active:
                self.set_status("轮到你了（显示推荐前三）")
                return
            legal = g.legal_actions()
            items = []
            for a in legal[:40]:
                items.append((action_desc(self.asp, a), lambda aa=a: self.play_action(aa)))
            self._set_actions(items)
            self.set_status("轮到你了")
            return
        a = self.model_action()
        self.logmsg(f"玩家{g.turn}（AI）出：{action_desc(self.asp, a)}")
        g.step(a)
        self._record()
        self.refresh()
        self._start_search()
        self._schedule(350)

    def play_action(self, a):
        g = self.game
        if self.ended or g is None or self.phase != "play" or g.turn != self.human_seat:
            return
        self.rec_active = False
        if a not in g.legal_actions():
            messagebox.showerror("非法动作", "这个动作当前不合法")
            return
        self.logmsg(f"你出：{action_desc(self.asp, a)}")
        g.step(a)
        self._record()
        self.picker.clear()
        self.refresh()
        self._start_search()
        self._schedule(150)

    def manual_play(self):
        g = self.game
        if self.ended or g is None or g.turn != self.human_seat:
            return
        m = self.picker.mask()
        if m == 0:
            messagebox.showinfo("提示", "请先选牌")
            return
        for a in g.legal_actions():
            if a != self.asp.n and int(self.asp.mask[a]) == m:
                self.play_action(a)
                return
        messagebox.showerror("非法牌型", "选的牌不构成合法牌型，或压不过上一手")

    def manual_pass(self):
        g = self.game
        if self.ended or g is None or g.turn != self.human_seat:
            return
        if self.asp.n in g.legal_actions():
            self.play_action(self.asp.n)
        else:
            messagebox.showinfo("提示", "现在不能过（自由出牌必须出牌）")


class AnalysisFrame(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=6)
        self.app = app
        self.asp = app.asp
        self.step = 0
        self.hand17 = 0
        self.bottom = 0
        self.grab = False
        self.landlord = 0
        self.session: AnalysisSession | None = None
        self.depth = tk.StringVar(value="64")
        self.remain_after_hand = TOTAL_COUNTS.copy()
        self._last_turn = -1
        self.history = []
        self.hcursor = -1
        self.ended = False
        self.last_best_actions = []
        self._analyze_after_id = None
        self._live_after_id = None
        self._live_token = 0
        self._search_gen = 0
        self._search_event = None
        self._search_queue = queue.Queue()
        self._search_thread = None
        self._search_poll = None

        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 2))
        ttk.Label(bar, text="游戏选项：", font=FONT_B).pack(side="left")
        ttk.Label(bar, text="牌局模式：", font=FONT_B).pack(side="left", padx=(4, 2))
        self.mode_var = tk.StringVar(value=self._mode_label(self.app.analysis_mode))
        self.mode_box = ttk.Combobox(bar, textvariable=self.mode_var, state="readonly",
                                     values=["常规模式", "不洗牌模式"], width=10)
        self.mode_box.pack(side="left", padx=2)
        self.mode_box.bind("<<ComboboxSelected>>", self._on_mode_change)
        self.btn_back = ttk.Button(bar, text="上一步", command=self.back_step)
        self.btn_back.pack(side="left", padx=4)
        ttk.Button(bar, text="重置 / 重新开始", command=self.reset_all).pack(side="left", padx=4)
        self.title = ttk.Label(self, text="分析模式", font=FONT_T)
        self.title.pack(anchor="w")

        # 步骤 0：输入 17 张
        self.f0 = ttk.LabelFrame(self, text="1) 输入你的 17 张手牌", padding=6)
        self.f0.pack(fill="x", pady=4)
        self.pick0 = CardPicker(self.f0, max_total=17, sort_desc=True, rank_desc=True)
        self.pick0.pack(fill="x")
        ttk.Button(self.f0, text="确认 17 张", command=self.confirm_hand).pack(anchor="w", pady=3)

        # 步骤 1：叫地主
        self.f1 = ttk.LabelFrame(self, text="2) 叫地主建议", padding=6)
        self.bid_lbl = ttk.Label(self.f1, text="—", font=FONT_B, justify="left")
        self.bid_lbl.pack(anchor="w")
        row = ttk.Frame(self.f1); row.pack(anchor="w", pady=3)
        ttk.Button(row, text="抢地主", command=lambda: self.confirm_grab(True)).pack(side="left", padx=2)
        ttk.Button(row, text="不抢", command=lambda: self.confirm_grab(False)).pack(side="left", padx=2)
        ttk.Button(row, text="精确模拟（rollout）", command=self.exact_bid).pack(side="left", padx=8)

        # 步骤 2：底牌 / 座位
        self.f2 = ttk.LabelFrame(self, text="3) 底牌 / 你的座位", padding=6)
        self.pick2 = CardPicker(self.f2, max_total=3, sort_desc=True, rank_desc=True)
        self.pick2.pack(fill="x")
        self.role_var = tk.StringVar(value="landlord")
        rr = ttk.Frame(self.f2); rr.pack(anchor="w", pady=3)
        ttk.Label(rr, text="你的身份：").pack(side="left")
        self.rb_ll = ttk.Radiobutton(rr, text="我抢到地主", variable=self.role_var, value="landlord")
        self.rb_r1 = ttk.Radiobutton(rr, text="我是地主下家", variable=self.role_var, value="next")
        self.rb_r2 = ttk.Radiobutton(rr, text="我是地主上家", variable=self.role_var, value="prev")
        for w in (self.rb_ll, self.rb_r1, self.rb_r2):
            w.pack(side="left", padx=4)
        ttk.Button(self.f2, text="确认底牌并开始", command=self.confirm_bottom).pack(anchor="w", pady=3)

        # 步骤 3：加倍
        self.f3 = ttk.LabelFrame(self, text="4) 加倍建议", padding=6)
        self.dbl_lbl = ttk.Label(self.f3, text="—", font=FONT_B, justify="left")
        self.dbl_lbl.pack(anchor="w")
        row = ttk.Frame(self.f3); row.pack(anchor="w", pady=3)
        ttk.Button(row, text="加倍", command=lambda: self.start_play(True)).pack(side="left", padx=2)
        ttk.Button(row, text="不加倍", command=lambda: self.start_play(False)).pack(side="left", padx=2)

        # 步骤 4：牌局
        self.f4 = ttk.Frame(self)
        self.counter_box = ttk.LabelFrame(self.f4, text="记牌器 · 未见牌（张）", padding=3)
        self.counter_box.pack(fill="x", pady=(0, 4))
        self.counter_labels = []
        for r in range(N_RANKS):
            self.counter_box.grid_columnconfigure(r, weight=1)
            ttk.Label(self.counter_box, text=RANK_NAMES[r],
                      font=("Microsoft YaHei UI", 14, "bold"),
                      anchor="center").grid(row=0, column=r, sticky="nsew", padx=1)
            lab = ttk.Label(self.counter_box, text="0",
                            font=("Consolas", 18, "bold"), anchor="center")
            lab.grid(row=1, column=r, sticky="nsew", padx=1)
            self.counter_labels.append(lab)
        top = ttk.Frame(self.f4); top.pack(fill="x")
        self.state_lbl = ttk.Label(top, text="—", font=FONT_TABLE, justify="left")
        self.state_lbl.pack(side="left", anchor="nw")
        self.lib_lbl = ttk.Label(top, text="—", font=FONT_SMALL, justify="left", wraplength=560)
        self.lib_lbl.pack(side="right", anchor="ne")
        self.camp_lbl = ttk.Label(self.f4, text="实时估计：地主 —  |  农民 —",
                                  font=("Microsoft YaHei UI", 13, "bold"))
        self.camp_lbl.pack(anchor="w", pady=(2, 0))
        self.rec = ScrollFrame(self.f4, height=110)
        self.rec.pack(fill="x", pady=3)
        self.pick4 = CardPicker(self.f4, max_total=20)
        self.pick4.pack(fill="x")
        row = ttk.Frame(self.f4); row.pack(fill="x", pady=2)
        self.btn_play4 = ttk.Button(row, text="出牌/记录出牌", command=self.play_selected)
        self.btn_play4.pack(side="left", padx=2)
        self.btn_pass4 = ttk.Button(row, text="过/记录过", command=self.record_pass)
        self.btn_pass4.pack(side="left", padx=2)
        self.btn_analyze = ttk.Button(row, text="分析/更新", command=self.analyze)
        self.btn_analyze.pack(side="left", padx=6)
        ttk.Label(row, text="深度").pack(side="left")
        ttk.Combobox(row, textvariable=self.depth, values=["8", "16", "32", "64", "128"],
                     state="readonly", width=5).pack(side="left")
        self.win_lbl = ttk.Label(self.f4, text="胜率：—", font=FONT_B)
        self.win_lbl.pack(anchor="w")
        self.msg = ttk.Label(self.f4, text="", font=FONT)
        self.msg.pack(anchor="w")
        self.turn_badge = ttk.Label(self.f4, text="", font=("Microsoft YaHei UI", 20, "bold"),
                                    foreground="#c62828")
        self.turn_badge.place(relx=1.0, rely=1.0, anchor="se", x=-8, y=-4)

        self._show_only(0)
        self._search_poll = self.after(100, self._poll_search)

    def _show_only(self, step):
        self.step = step
        for f in (self.f0, self.f1, self.f2, self.f3, self.f4):
            f.pack_forget()
        frames = {0: self.f0, 1: self.f1, 2: self.f2, 3: self.f3, 4: self.f4}
        frames[step].pack(fill="both", expand=True, pady=4)
        self._sync_back_button()

    def _sync_back_button(self):
        if not hasattr(self, "btn_back"):
            return
        if self.step in (1, 2, 3):
            ok = True
        elif self.step == 4:
            ok = (not self.ended and self.session is not None
                  and self.session.winner < 0 and len(self.history) <= 1)
        else:
            ok = False
        self.btn_back.configure(state=("normal" if ok else "disabled"))

    # ---------- 0.2s anytime 搜索 ----------
    def _cancel_search(self):
        self._search_gen += 1
        ev = self._search_event
        self._search_event = None
        if ev is not None:
            ev.set()
        self._cancel_analyze()
        self._cancel_live()

    def _start_search(self):
        self._cancel_search()
        if self.step == 1 and self.hand17:
            self._start_bid_search()
        elif (self.step == 4 and self.session is not None
              and not self.ended and self.session.winner < 0):
            sess = clone_session(self.session)
            if sess.turn == sess.human_seat:
                self._start_top3_search(sess)
            else:
                self._start_camp_search(sess)

    def _start_bid_search(self):
        ev = threading.Event()
        self._search_event = ev
        gen = self._search_gen
        q = self._search_queue
        hand = self.hand17
        model = self.app.analysis_model
        device = self.app.device
        feat = self.app.action_feat
        asp = self.asp

        def run():
            n = 8
            try:
                while not ev.is_set() and n <= 512:
                    p_ll, p_f = bid_rollout(hand, model, device, feat,
                                            samples=n, seed=random.randrange(1 << 30), asp=asp)
                    q.put(("bid", gen, p_ll, p_f, n))
                    n = min(512, n * 2)
                    if ev.wait(0.2):
                        break
            except Exception as e:
                q.put(("error", gen, str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _start_top3_search(self, sess):
        ev = threading.Event()
        self._search_event = ev
        gen = self._search_gen
        q = self._search_queue
        model = self.app.analysis_model
        device = self.app.device
        feat = self.app.action_feat
        depth = int(self.depth.get())

        def run():
            try:
                def cb(best, pol, samples, elapsed):
                    q.put(("top3", gen, best, pol, samples, elapsed))
                sess.candidates(model, device, feat, topk=3,
                                samples=depth, max_samples=512, time_budget_ms=6000,
                                progress_cb=cb, cancel_event=ev, publish_ms=100)
            except Exception as e:
                q.put(("error", gen, str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _start_camp_search(self, sess):
        ev = threading.Event()
        self._search_event = ev
        gen = self._search_gen
        q = self._search_queue
        model = self.app.analysis_model
        device = self.app.device
        feat = self.app.action_feat

        def run():
            n = 8
            try:
                while not ev.is_set() and n <= 512:
                    ll, ff = sess.camp_value(model, device, feat, samples=n)
                    q.put(("camp", gen, ll, ff, n))
                    n = min(512, n * 2)
                    if ev.wait(0.2):
                        break
            except Exception as e:
                q.put(("error", gen, str(e)))

        threading.Thread(target=run, daemon=True).start()

    def _poll_search(self):
        if not self.winfo_exists():
            return
        while True:
            try:
                item = self._search_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            gen = item[1]
            if gen != self._search_gen:
                continue
            if kind == "top3" and self.step == 4 and self.session is not None:
                _, _, best, pol, samples, elapsed = item
                self.last_best_actions = [a for a, _ in best]
                self._set_rec(self._items_for_human(best))
                self.win_lbl.configure(
                    text=f"实时前三选（{samples} 次采样 / {elapsed:.0f} ms）最高 {pol:.1%}")
                s = self.session
                if s.human_seat == s.landlord:
                    self._set_camp(pol, 1.0 - pol)
                else:
                    self._set_camp(1.0 - pol, pol)
                self.msg.configure(text="每 0.2s 更新，采样越深越准；手动可选任意合法牌")
            elif kind == "camp" and self.step == 4:
                _, _, ll, ff, samples = item
                self._set_camp(ll, ff)
            elif kind == "bid" and self.step == 1:
                _, _, p_ll, p_f, samples = item
                self._render_bid(p_ll, p_f, samples)
            elif kind == "error":
                self.msg.configure(text=f"实时搜索失败：{item[2]}")
        self._search_poll = self.after(100, self._poll_search)

    def back_step(self):
        """设置阶段回退：保留已选牌/底牌/座位，只丢弃下游状态。"""
        if self.step == 0:
            messagebox.showinfo("提示", "已经是第一步")
            return
        self._cancel_search()
        self.camp_lbl.configure(text="实时估计：地主 —  |  农民 —")
        if self.step == 1:
            # 回到 17 张输入；保留 pick0 的牌，清掉底牌和下游状态
            self.session = None
            self.history = []
            self.hcursor = -1
            self.ended = False
            self.grab = False
            self.pick2.clear()
            self.pick2.set_limits(TOTAL_COUNTS, 3)
            self.pick4.clear()
            self.remain_after_hand = TOTAL_COUNTS.copy()
            for rb in (self.rb_ll, self.rb_r1, self.rb_r2):
                rb.configure(state="normal")
            self.bid_lbl.configure(text="—")
            self.dbl_lbl.configure(text="—")
            self.title.configure(text="分析模式")
            self._show_only(0)
        elif self.step == 2:
            # 回到叫地主；保留已选底牌，若改抢/不抢可直接继续
            self.session = None
            self.history = []
            self.hcursor = -1
            self.ended = False
            self.pick4.clear()
            self.dbl_lbl.configure(text="—")
            self._show_only(1)
        elif self.step == 3:
            # 回到底牌；保留底牌和座位
            self.session = None
            self.history = []
            self.hcursor = -1
            self.ended = False
            self.pick4.clear()
            self.dbl_lbl.configure(text="—")
            self.title.configure(text="分析模式")
            self._show_only(2)
        elif self.step == 4:
            if self.ended or (self.session is not None and self.session.winner >= 0):
                messagebox.showinfo("提示", "牌局已结束，不能回到设置阶段")
                return
            if len(self.history) > 1:
                messagebox.showinfo(
                    "提示",
                    "牌局已经开始，请用「上一回合」/「撤销」回退出牌；\n"
                    "如果要改底牌/座位，请用「重置 / 重新开始」。")
                return
            self.pick4.clear()
            self.title.configure(text="分析模式")
            self._show_only(3)

    @staticmethod
    def _mode_label(mode):
        return "不洗牌模式" if mode == "noshuffle" else "常规模式"

    def _on_mode_change(self, _event=None):
        mode = "noshuffle" if self.mode_var.get().startswith("不洗牌") else "regular"
        self.app.set_analysis_mode(mode)
        self._cancel_search()
        if mode == "noshuffle" and self.app.model_noshuffle is None:
            self.title.configure(text="分析模式（不洗牌模型缺失，按常规分析）")
        else:
            self.title.configure(text=f"分析模式（{self._mode_label(mode)}）")

    def reset_all(self):
        self._cancel_search()
        self.last_best_actions = []
        self.camp_lbl.configure(text="实时估计：地主 —  |  农民 —")
        self.pick0.clear()
        self.pick0.set_limits(TOTAL_COUNTS, 17)
        self.pick2.clear()
        self.pick2.set_limits(TOTAL_COUNTS, 3)
        self.pick4.clear()
        self.pick4.set_limits(TOTAL_COUNTS, 20)
        self.hand17 = 0
        self.bottom = 0
        self.session = None
        self.grab = False
        self.landlord = 0
        self.remain_after_hand = TOTAL_COUNTS.copy()
        self._last_turn = -1
        self.history = []
        self.hcursor = -1
        self.ended = False
        self.last_best_actions = []
        self.role_var.set("landlord")
        for rb in (self.rb_ll, self.rb_r1, self.rb_r2):
            rb.configure(state="normal")
        self.rec.clear()
        self.win_lbl.configure(text="胜率：—")
        self.msg.configure(text="")
        self.bid_lbl.configure(text="—")
        self.dbl_lbl.configure(text="—")
        self.title.configure(text="分析模式")
        self._show_only(0)

    def confirm_hand(self):
        n = self.pick0.total()
        if n != 17:
            messagebox.showerror("张数不对", f"需要 17 张，当前 {n} 张")
            return
        self._cancel_search()
        self.camp_lbl.configure(text="实时估计：地主 —  |  农民 —")
        self.hand17 = self.pick0.mask()
        self.grab = False
        self.session = None
        self.history = []
        self.hcursor = -1
        self.ended = False
        self.pick4.clear()
        self.pick2.clear()
        self.dbl_lbl.configure(text="—")
        self.role_var.set("landlord")
        for rb in (self.rb_ll, self.rb_r1, self.rb_r2):
            rb.configure(state="normal")
        self.remain_after_hand = np.maximum(
            TOTAL_COUNTS.astype(np.int16) - mask_to_counts(self.hand17).astype(np.int16), 0
        ).astype(np.int8)
        self.pick2.set_limits(self.remain_after_hand, 3)
        if self.app.analysis_mode == "noshuffle" and self.app.analysis_model is not None:
            # 不洗牌模式必须用不洗牌专家自己的参数估胜率，不能借用常规 bid net。
            p_ll, p_f = bid_rollout(self.hand17, self.app.analysis_model, self.app.device,
                                    self.app.action_feat, samples=32,
                                    seed=random.randrange(1 << 30), asp=self.asp)
        else:
            p_ll, p_f, _ = bid_advice(self.app.bid_net, self.hand17, self.app.device)
        self._bid_cache = (p_ll, p_f)
        self._render_bid(p_ll, p_f)
        self._show_only(1)
        self._start_search()

    def _render_bid(self, p_ll, p_f, samples=None):
        extra = f"（实时 {samples} 次采样）" if samples is not None else ""
        if self.app.analysis_mode == "noshuffle":
            rec = "建议抢地主" if should_grab_noshuffle(p_ll) else "建议不抢，当地主下家/上家农民"
            rule = "不洗牌模式：地主胜率 >75% 才推荐抢"
        else:
            rec = "建议抢地主" if should_grab(p_ll, p_f) else "建议不抢，当地主下家/上家农民"
            rule = "地主胜率>农民 / 地主>70% / 低于农民不超5%"
        self.bid_lbl.configure(
            text=f"你的 17 张：{mask_str(self.hand17)}\n"
                 f"当地主胜率：{p_ll:.1%}    当农民胜率：{p_f:.1%}{extra}\n"
                 f"规则：{rule} → {rec}")

    def exact_bid(self):
        self.bid_lbl.configure(text="实时搜索中…（每 0.2s 更新）")
        self._start_search()

    def confirm_grab(self, grab: bool):
        self._cancel_search()
        self.camp_lbl.configure(text="实时估计：地主 —  |  农民 —")
        self.grab = grab
        self.role_var.set("landlord" if grab else "next")
        self.rb_ll.configure(state="disabled" if not grab else "normal")
        self.rb_r1.configure(state="disabled" if grab else "normal")
        self.rb_r2.configure(state="disabled" if grab else "normal")
        self.session = None
        self.history = []
        self.hcursor = -1
        self.ended = False
        self.pick4.clear()
        self.dbl_lbl.configure(text="—")
        self.title.configure(text="分析模式")
        self.pick2.set_limits(self.remain_after_hand, 3)
        self._show_only(2)

    def confirm_bottom(self):
        if self.pick2.total() != 3:
            messagebox.showerror("张数不对", "底牌需要 3 张")
            return
        self.bottom = self.pick2.mask()
        if self.grab:
            self.landlord = 0
        else:
            role = self.role_var.get()
            self.landlord = 2 if role == "next" else 1
        if np.any(self.pick2.counts > self.remain_after_hand):
            messagebox.showerror("底牌不合法", "底牌含有已被你的 17 张手牌耗尽的点数")
            return
        self._cancel_search()
        self.camp_lbl.configure(text="实时估计：地主 —  |  农民 —")
        self.session = AnalysisSession.start(self.hand17, self.grab, self.bottom, self.landlord)
        self.history = []
        self.hcursor = -1
        self.ended = False
        self.pick4.clear()
        self._last_turn = -1
        self.dbl_lbl.configure(text="计算加倍建议…")
        self._show_only(3)
        self.update_idletasks()
        wr = self.session.policy_value(self.app.analysis_model, self.app.device, self.app.action_feat,
                                       samples=48)
        rec = "建议加倍" if wr > 0.70 else "不建议加倍"
        self.dbl_lbl.configure(text=f"你的初始胜率估计：{wr:.1%}\n"
                                    f"规则：胜率 > 70% 推荐加倍 → {rec}")

    def start_play(self, doubled: bool):
        self.history = [clone_session(self.session)]
        self.hcursor = 0
        self.ended = False
        self._show_only(4)
        self.title.configure(text=f"分析模式（{'已加倍' if doubled else '未加倍'}）")
        self._refresh_state()
        self.camp_lbl.configure(text="实时估计：计算中…")
        self._start_search()

    # ---------- 历史 / 撤销 ----------
    def _cancel_analyze(self):
        if self._analyze_after_id is not None:
            try:
                self.after_cancel(self._analyze_after_id)
            except Exception:
                pass
            self._analyze_after_id = None

    def _schedule_analyze(self, ms=100):
        self._cancel_analyze()
        self._analyze_after_id = self.after(ms, self.analyze)

    def _cancel_live(self):
        if self._live_after_id is not None:
            try:
                self.after_cancel(self._live_after_id)
            except Exception:
                pass
            self._live_after_id = None
        self._live_token += 1

    def _schedule_live(self, ms=80):
        self._cancel_live()
        token = self._live_token
        self._live_after_id = self.after(ms, lambda: self._run_live(token))

    def _run_live(self, token):
        self._live_after_id = None
        if token != self._live_token:
            return
        s = self.session
        if s is None or self.ended:
            return
        try:
            ll, ff = s.camp_value(self.app.analysis_model, self.app.device, self.app.action_feat,
                                  samples=32)
        except Exception:
            return
        if token != self._live_token or self.session is not s:
            return
        self._set_camp(ll, ff)

    def _set_camp(self, ll, ff):
        s = self.session
        if s is None:
            return
        if s.winner >= 0:
            self.camp_lbl.configure(
                text=f"牌局结束：{'地主' if s.winner == s.landlord else '农民'}胜")
            return
        you = "地主" if s.human_seat == s.landlord else "农民"
        self.camp_lbl.configure(
            text=f"实时估计：地主 {ll:.1%}  |  农民 {ff:.1%}  （你是{you}）")

    def on_close(self):
        self._cancel_search()
        if self._search_poll is not None:
            try:
                self.after_cancel(self._search_poll)
            except Exception:
                pass
            self._search_poll = None

    def _record(self):
        if self.session is None:
            return
        self.history = self.history[:self.hcursor + 1]
        self.history.append(clone_session(self.session))
        self.hcursor = len(self.history) - 1
        self._sync_back_button()

    def _restore(self, idx):
        if not (0 <= idx < len(self.history)):
            return
        self.session = clone_session(self.history[idx])
        self.hcursor = idx
        self.ended = False
        self.pick4.clear()
        self._last_turn = -1
        self.rec.clear()
        self.last_best_actions = []
        self.win_lbl.configure(text="胜率：—")
        self.camp_lbl.configure(text="实时估计：计算中…")
        self._show_only(4)
        self._refresh_state()
        self._start_search()

    def prev_turn(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束，请重置")
            return
        if not self.history:
            messagebox.showinfo("提示", "牌局还没开始")
            return
        if self.hcursor <= 0:
            messagebox.showinfo("提示", "已经是牌局开始")
            return
        self.pick4.clear()
        self._restore(self.hcursor - 1)

    def next_turn(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束，请重置")
            return
        if not self.history:
            messagebox.showinfo("提示", "牌局还没开始")
            return
        if self.hcursor >= len(self.history) - 1:
            messagebox.showinfo("提示", "没有下一回合")
            return
        self._restore(self.hcursor + 1)

    def undo(self):
        if self.ended:
            messagebox.showinfo("提示", "牌局已结束，请重置")
            return
        if self.step == 0:
            if self.pick0.total() > 0:
                self.pick0.undo()
            return
        if self.step in (1, 2, 3):
            self.back_step()
            return
        if self.pick4.total() > 0:
            self.pick4.clear()
            self.msg.configure(text="已清空本次选牌")
            return
        self.prev_turn()

    def reset_play(self):
        if not self.history:
            self.reset_all()
            return
        first = clone_session(self.history[0])
        self.history = [first]
        self.hcursor = 0
        self.session = clone_session(first)
        self.ended = False
        self.pick4.clear()
        self._last_turn = -1
        self.rec.clear()
        self.win_lbl.configure(text="胜率：—")
        self.camp_lbl.configure(text="实时估计：计算中…")
        self.title.configure(text="分析模式")
        self._show_only(4)
        self._refresh_state()
        self._start_search()

    def end_game(self):
        if self.session is None:
            messagebox.showinfo("提示", "牌局还没开始")
            return
        self._cancel_search()
        self.ended = True
        self.pick4.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
        if self.session.winner >= 0:
            ll = 1.0 if self.session.winner == self.session.landlord else 0.0
            self._set_camp(ll, 1.0 - ll)
        else:
            self.camp_lbl.configure(text="实时估计：—（已结束）")
        self.rec.clear()
        self.win_lbl.configure(text="胜率：—")
        self.msg.configure(text="牌局已手动结束；可点「牌局重置」回到开始，或「重置 / 重新开始」重新输入")
        self.title.configure(text="分析模式（已结束）")
        self._refresh_state()

    def ribbon(self, action):
        if action == "reset":
            self.reset_play()
        elif action == "end":
            self.end_game()
        elif action == "undo":
            self.undo()
        elif action == "prev":
            if self.step <= 3 or (self.step == 4 and len(self.history) <= 1
                                   and not self.ended):
                self.back_step()
            else:
                self.prev_turn()
        elif action == "next":
            self.next_turn()

    def _refresh_state(self):
        s = self.session
        if s is None:
            return
        parts = []
        for p in range(3):
            role = "地主" if p == s.landlord else "农民"
            you = "（你）" if p == s.human_seat else ""
            mark = "←" if p == s.turn and s.winner < 0 and not self.ended else ""
            parts.append(f"P{p}{you}{role}{s.hand_sizes[p]}张{mark}")
        lines = ["  ".join(parts)]
        lines.append(f"底牌 {mask_str(s.bottom)}   上一手 "
                     f"{action_desc(self.asp, s.last_act) if s.last_act >= 0 else '无'}")
        lines.append(f"你的手牌 {mask_str(s.human_hand)}")
        if s.winner >= 0:
            lines.append(f"*** 玩家{s.winner} 获胜 ***")
        self.state_lbl.configure(text="\n".join(lines))
        unseen = s.unseen_counts()
        for r, lab in enumerate(self.counter_labels):
            n = int(unseen[r])
            lab.configure(text=str(n), foreground=("#c62828" if n == 0 else "#111"))
        self.lib_lbl.configure(text="已出：\n" +
                               "\n".join(f"P{p}: {played_str(s.played[p])}" for p in range(3)))
        if self.ended:
            self.msg.configure(text="牌局已手动结束")
        else:
            self.msg.configure(text=f"轮到玩家{s.turn}" if s.winner < 0 else "牌局结束")
        self._update_pick4_limits()
        self._sync_buttons()
        self._sync_back_button()
        if self.ended or s.winner >= 0:
            self.turn_badge.configure(text="牌局结束", font=("Microsoft YaHei UI", 14, "bold"),
                                      foreground="#555")
        elif s.turn == s.human_seat:
            self.turn_badge.configure(text="该你出牌！！！",
                                      font=("Microsoft YaHei UI", 24, "bold"),
                                      foreground="#c62828")
        else:
            rel = (int(s.turn) - s.human_seat) % 3
            side = "下家" if rel == 1 else "上家"
            self.turn_badge.configure(text=f"该你的{side}出牌",
                                      font=("Microsoft YaHei UI", 20, "bold"),
                                      foreground="#444")

    def _sync_buttons(self):
        s = self.session
        if s is None or self.ended or s.winner >= 0:
            for b in (self.btn_play4, self.btn_pass4, self.btn_analyze):
                b.configure(state="disabled")
            return
        if s.turn == s.human_seat:
            self.btn_play4.configure(state="normal")
            self.btn_analyze.configure(state="normal")
            self.btn_pass4.configure(
                state=("normal" if self.asp.n in s.legal_actions() else "disabled"))
        else:
            self.btn_play4.configure(state="normal")
            self.btn_analyze.configure(state="disabled")
            self.btn_pass4.configure(
                state=("normal" if s.last_act >= 0 else "disabled"))

    def _update_pick4_limits(self):
        s = self.session
        if s is None or self.ended:
            self.pick4.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
            return
        if s.winner >= 0:
            self.pick4.set_limits(np.zeros(N_RANKS, dtype=np.int8), 0)
            return
        if s.turn != self._last_turn:
            self.pick4.clear()
            self._last_turn = s.turn
        if s.turn == s.human_seat:
            lim = mask_to_counts(s.human_hand)
            mt = max(1, int(lim.sum()))
        else:
            lim = s.unseen_counts()
            mt = max(1, int(s.hand_sizes[s.turn]))
        self.pick4.set_limits(lim, mt)

    def _items_for_human(self, best):
        items = []
        for a, v in best:
            items.append((f"{action_desc(self.asp, a)}   胜率 {v:.1%}",
                          lambda aa=a: self.apply_human(aa)))
        return items

    def _set_rec(self, items):
        self.rec.clear()
        if not items:
            ttk.Label(self.rec.inner, text="（暂无）").pack(anchor="w")
            return
        for text, cb in items:
            ttk.Button(self.rec.inner, text=text, command=cb).pack(fill="x", pady=1)

    def analyze(self):
        if not self.winfo_exists():
            return
        s = self.session
        if self.ended or s is None or s.winner >= 0:
            return
        self.msg.configure(text="实时搜索中…（每 0.2s 更新）")
        self._start_search()

    def apply_human(self, a):
        s = self.session
        if self.ended or s is None or s.turn != s.human_seat:
            return
        try:
            s.apply(s.human_seat, a)
        except Exception as e:
            messagebox.showerror("非法动作", str(e))
            return
        self._record()
        self.pick4.clear()
        self._refresh_state()
        if a == self.asp.n:
            self.msg.configure(text="你选择过")
        if s.winner >= 0:
            self._set_rec([])
            self._cancel_search()
            ll = 1.0 if s.winner == s.landlord else 0.0
            self._set_camp(ll, 1.0 - ll)
            return
        self._start_search()

    def play_selected(self):
        s = self.session
        if self.ended or s is None or s.winner >= 0:
            return
        m = self.pick4.mask()
        if m == 0:
            messagebox.showinfo("提示", "请先选牌")
            return
        if s.turn == s.human_seat:
            for a in s.legal_actions():
                if a != self.asp.n and int(self.asp.mask[a]) == m:
                    self.apply_human(a)
                    return
            messagebox.showerror("非法牌型", "不是合法牌型或压不过上一手")
        else:
            # 记录对手出牌：按 mask 找动作并校验
            idx = np.nonzero(self.asp.mask == np.int64(m))[0]
            if len(idx) == 0:
                messagebox.showerror("非法牌型", "这组牌不是合法牌型")
                return
            a = int(idx[0])
            if not s.is_legal_for(s.turn, a):
                messagebox.showerror("非法动作", "压不过上一手，或没有这么多牌")
                return
            p = s.turn
            s.apply(p, a)
            self._record()
            self.pick4.clear()
            self._refresh_state()
            if s.winner >= 0:
                ll = 1.0 if s.winner == s.landlord else 0.0
                self._set_camp(ll, 1.0 - ll)
            self._start_search()

    def record_pass(self):
        s = self.session
        if self.ended or s is None or s.winner >= 0:
            return
        if s.turn == s.human_seat:
            if self.asp.n not in s.legal_actions():
                messagebox.showinfo("提示", "自由出牌不能过")
                return
            self.apply_human(self.asp.n)
        else:
            if s.last_act < 0:
                messagebox.showinfo("提示", "自由出牌不能过")
                return
            p = s.turn
            s.apply(p, self.asp.n)
            self._record()
            self._refresh_state()
            self.msg.configure(text=f"记录：玩家{p}过")
            self._start_search()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("斗地主 AI — 自对弈模型 / 叫地主 / 加倍 / 分析")
        self.geometry("1440x960")
        self.minsize(1240, 840)
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", font=FONT)
        style.configure("TButton", font=FONT, padding=(6, 3))
        style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 13, "bold"), padding=(14, 5))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.asp = get_action_space()
        self.model = None
        self.model_regular = None
        self.model_noshuffle = None
        self.analysis_model = None
        self.analysis_mode = "regular"
        self.bid_net = None
        self.action_feat = None
        splash = tk.Toplevel(self)
        splash.title("初始化")
        splash.geometry("520x150+500+300")
        splash.transient(self)
        ttk.Label(splash, text="正在加载模型并预热推理引擎…\n首次启动约 10–20 秒",
                  font=FONT_B, justify="center").pack(expand=True)
        splash.update()
        self._load_models()
        try:
            self._warmup()
        except Exception as e:
            print("warmup failed", e)
        splash.destroy()

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=4, pady=(2, 4))
        self.human_tab = HumanFrame(self.nb, self)
        self.analysis_tab = AnalysisFrame(self.nb, self)
        self.nb.add(self.human_tab, text="人机对弈")
        self.nb.add(self.analysis_tab, text="分析模式")

        ribbon = tk.Frame(self, bg="#143d2e", height=58)
        ribbon.pack(fill="x", side="top", before=self.nb)
        ribbon.pack_propagate(False)
        tk.Label(ribbon, text="斗地主 AI", bg="#143d2e", fg="white",
                 font=("Microsoft YaHei UI", 20, "bold")).pack(side="left", padx=(16, 10))
        tk.Label(ribbon, text="5.61M 自对弈 · 0.2s 实时 · 叫地主 / 加倍 / 前三选",
                 bg="#143d2e", fg="#cfe8d8",
                 font=("Microsoft YaHei UI", 12)).pack(side="left")
        def rb(text, cmd):
            return tk.Button(ribbon, text=text, command=cmd, bg="#2e6b52", fg="white",
                             activebackground="#3d8a6a", activeforeground="white",
                             relief="flat", bd=0, padx=12, pady=6,
                             font=("Microsoft YaHei UI", 11, "bold"))
        rb("说明", self._show_help).pack(side="right", padx=(4, 14))
        rb("牌局结束", lambda: self._ribbon_action("end")).pack(side="right", padx=4)
        rb("撤销", lambda: self._ribbon_action("undo")).pack(side="right", padx=4)
        rb("下一回合", lambda: self._ribbon_action("next")).pack(side="right", padx=4)
        rb("上一回合", lambda: self._ribbon_action("prev")).pack(side="right", padx=4)
        rb("牌局重置", lambda: self._ribbon_action("reset")).pack(side="right", padx=4)
        self.bind("<Control-z>", lambda e: self._ribbon_action("undo"))
        self.bind("<Control-Left>", lambda e: self._ribbon_action("prev"))
        self.bind("<Control-Right>", lambda e: self._ribbon_action("next"))
        self.human_tab.logmsg("提示：顶部 ribbon 支持 牌局重置 / 上一回合 / 下一回合 / 撤销 / 牌局结束。")

    def set_analysis_mode(self, mode):
        self.analysis_mode = "noshuffle" if mode == "noshuffle" else "regular"
        if self.analysis_mode == "noshuffle" and self.model_noshuffle is not None:
            self.analysis_model = self.model_noshuffle
        else:
            self.analysis_model = self.model_regular

    def _ribbon_action(self, action):
        idx = self.nb.index(self.nb.select())
        frame = self.human_tab if idx == 0 else self.analysis_tab
        frame.ribbon(action)

    def _show_help(self):
        messagebox.showinfo(
            "使用说明",
            "人机对弈：新开一局 → 叫地主建议 → 抢/不抢 → 底牌 → 加倍建议 → 出牌。\n"
            "分析模式：输入 17 张 → 叫地主建议 → 底牌/座位 → 加倍建议 → 记录出牌。\n"
            "设置阶段点「上一步」可回到上一阶段，已选的牌/底牌/座位会保留。\n\n"
            "分析模式中：\n"
            "· 顶部“牌局模式”可选常规/不洗牌；重置后保持；\n"
            "· 点上方大号点数加牌；点手牌栏的大牌可放回一张；\n"
            "· 每个点数下方显示剩余可选的张数；\n"
            "· 最多 17 张；底牌只能从剩余牌中选 3 张；\n"
            "· 轮到你时自动给前三选及胜率。\n\n"
            "顶部 ribbon：\n"
            "· 牌局重置：人机=新开一局；分析=回到本局开始；\n"
            "· 上一回合 / 下一回合：按动作回退/前进；\n"
            "· 撤销（Ctrl+Z）：先清空本次选牌，没有选牌则回退一步；\n"
            "· 牌局结束：手动结束本局。\n"
            "分析模式「重置 / 重新开始」会清空全部输入。",
        )

    def _warmup(self):
        from .reference import Game
        from .rollout import run_policy
        from .analysis import AnalysisSession
        eng = BatchEngine(8, seed=1, auto_landlord=False)
        eng.assign_landlords(np.arange(8, dtype=np.int32), np.zeros(8, dtype=np.int8))
        run_policy(eng, self.model, self.device, self.action_feat, max_steps=120)
        gg = Game.deal(seed=123)
        s = AnalysisSession.start(gg.hands[0], True, gg.bottom, 0)
        s.candidates(self.model, self.device, self.action_feat, topk=1,
                     time_budget_ms=300, max_samples=4)

    def destroy(self):
        for f in (getattr(self, "human_tab", None), getattr(self, "analysis_tab", None)):
            if f is not None:
                try:
                    f.on_close()
                except Exception:
                    pass
        super().destroy()

    def _load_models(self):
        model_path = Path(os.environ.get("DOUDIZHU_MODEL", str(ROOT / "checkpoints" / "best_model.pt")))
        bid_path = Path(os.environ.get("DOUDIZHU_BID", str(ROOT / "checkpoints" / "bid_net.pt")))
        if not model_path.exists():
            messagebox.showerror("缺少模型", f"找不到 {model_path}")
            raise SystemExit(1)
        self.model_regular, _ = load_checkpoint(str(model_path), map_location=self.device)
        self.model_regular.to(self.device).eval()
        self.model = self.model_regular
        self.analysis_model = self.model_regular
        ns_path = Path(os.environ.get("DOUDIZHU_MODEL_NOSHUFFLE",
                                      str(ROOT / "checkpoints" / "best_model_noshuffle.pt")))
        if ns_path.exists():
            self.model_noshuffle, _ = load_checkpoint(str(ns_path), map_location=self.device)
            self.model_noshuffle.to(self.device).eval()
            print("no-shuffle model loaded", ns_path)
        else:
            print("no-shuffle model not found; mode falls back to regular")
        self.action_feat = torch.as_tensor(make_action_features(self.asp), dtype=torch.float32,
                                           device=self.device)
        if bid_path.exists():
            self.bid_net, _ = load_bid_net(str(bid_path), map_location=self.device)
            self.bid_net.to(self.device).eval()
            print("bid net loaded", bid_path)
        else:
            print("bid_net.pt not found; bidding falls back to 50%")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
