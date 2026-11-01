"""纯 Python 参考实现：用于交互对局、规则测试和 Numba 引擎的交叉验证。"""
from __future__ import annotations

import numpy as np

from .actions import (
    ActionSpace, BOMB, FULL_DECK, N_RANKS, ROCKET, counts_to_mask, mask_to_counts,
)

_ASP: ActionSpace | None = None


def canonicalize_mask(mask: int) -> int:
    """把任意花色位表示的牌重新规范到每个点数最低的若干张。"""
    return counts_to_mask(mask_to_counts(mask))


def get_action_space() -> ActionSpace:
    global _ASP
    if _ASP is None:
        _ASP = ActionSpace()
    return _ASP


class Game:
    """单局斗地主（随机地主，省略叫牌），出牌阶段完整规则。"""

    def __init__(self, seed=None, landlord=None, hands=None, bottom=None):
        self.asp = get_action_space()
        self.pass_id = self.asp.n
        if hands is None:
            rng = np.random.default_rng(seed)
            deck = np.array([i // 4 for i in range(52)] + [13, 14], dtype=np.int64)
            rng.shuffle(deck)
            cnts = np.zeros((3, N_RANKS), dtype=np.int8)
            for i in range(51):
                cnts[i % 3, int(deck[i])] += 1
            bcnt = np.zeros(N_RANKS, dtype=np.int8)
            for i in range(51, 54):
                bcnt[int(deck[i])] += 1
            self.hands = [counts_to_mask(cnts[p]) for p in range(3)]
            self.bottom = counts_to_mask(bcnt)
            if landlord is None:
                landlord = int(rng.integers(0, 3))
        else:
            self.hands = list(hands)
            self.bottom = bottom or 0
            if landlord is None:
                landlord = 0
        self.landlord = int(landlord)
        self.hands[self.landlord] = counts_to_mask(
            mask_to_counts(self.hands[self.landlord]) + mask_to_counts(self.bottom)
        )
        self.played = np.zeros((3, N_RANKS), dtype=np.int8)
        self.turn = self.landlord
        self.last_act = -1
        self.last_player = -1
        self.passes = 0
        self.winner = -1
        self.history: list[tuple[int, int]] = []

    @classmethod
    def deal(cls, seed=None):
        """只发 17/17/17 + 底牌，不指定地主（用于叫地主阶段）。"""
        g = cls.__new__(cls)
        g.asp = get_action_space()
        g.pass_id = g.asp.n
        rng = np.random.default_rng(seed)
        deck = np.array([i // 4 for i in range(52)] + [13, 14], dtype=np.int64)
        rng.shuffle(deck)
        cnts = np.zeros((3, N_RANKS), dtype=np.int8)
        for i in range(51):
            cnts[i % 3, int(deck[i])] += 1
        bcnt = np.zeros(N_RANKS, dtype=np.int8)
        for i in range(51, 54):
            bcnt[int(deck[i])] += 1
        g.hands = [counts_to_mask(cnts[p]) for p in range(3)]
        g.bottom = counts_to_mask(bcnt)
        g.landlord = -1
        g.played = np.zeros((3, N_RANKS), dtype=np.int8)
        g.turn = -1
        g.last_act = -1
        g.last_player = -1
        g.passes = 0
        g.winner = -1
        g.history = []
        return g

    def assign_landlord(self, landlord: int):
        if self.landlord >= 0:
            raise RuntimeError("landlord already assigned")
        self.landlord = int(landlord)
        self.hands[self.landlord] = counts_to_mask(
            mask_to_counts(self.hands[self.landlord]) + mask_to_counts(self.bottom)
        )
        self.turn = self.landlord
        self.last_act = -1
        self.last_player = -1
        self.passes = 0
        self.winner = -1
        self.history = []

    @classmethod
    def from_state(cls, hands, bottom, landlord, turn, last_act=-1, last_player=-1,
                   passes=0, played=None):
        g = cls.__new__(cls)
        g.asp = get_action_space()
        g.pass_id = g.asp.n
        g.hands = [int(x) for x in hands]
        g.bottom = int(bottom)
        g.landlord = int(landlord)
        g.turn = int(turn)
        g.last_act = int(last_act)
        g.last_player = int(last_player)
        g.passes = int(passes)
        g.winner = -1
        g.played = np.zeros((3, N_RANKS), dtype=np.int8) if played is None else np.array(played, dtype=np.int8)
        g.history = []
        return g

    # ---------- 查询 ----------
    def card_count(self, p: int) -> int:
        return int(self.hands[p]).bit_count()

    def legal_actions(self) -> list[int]:
        hand = self.hands[self.turn]
        asp = self.asp
        out: list[int] = []
        if self.last_act < 0:
            m = asp.mask
            for i in range(asp.n):
                if (int(m[i]) & hand) == int(m[i]):
                    out.append(i)
            return out
        la = self.last_act
        t = int(asp.typ[la])
        c = int(asp.chain[la])
        r = int(asp.rank[la])
        if t == ROCKET:
            return [self.pass_id]
        if t == BOMB:
            s = int(asp.group_start[BOMB * 16])
            e = int(asp.group_end[BOMB * 16])
            for i in range(s, e):
                if int(asp.rank[i]) > r and (int(asp.mask[i]) & hand) == int(asp.mask[i]):
                    out.append(i)
            rid = int(asp.group_start[ROCKET * 16])
            if (int(asp.mask[rid]) & hand) == int(asp.mask[rid]):
                out.append(rid)
            out.append(self.pass_id)
            return out
        key = t * 16 + c
        s = int(asp.group_start[key])
        e = int(asp.group_end[key])
        for i in range(s, e):
            if int(asp.rank[i]) > r and (int(asp.mask[i]) & hand) == int(asp.mask[i]):
                out.append(i)
        bs = int(asp.group_start[BOMB * 16])
        be = int(asp.group_end[BOMB * 16])
        for i in range(bs, be):
            if int(asp.mask[i]) & hand == int(asp.mask[i]):
                out.append(i)
        rid = int(asp.group_start[ROCKET * 16])
        if (int(asp.mask[rid]) & hand) == int(asp.mask[rid]):
            out.append(rid)
        out.append(self.pass_id)
        return out

    def is_legal(self, action: int) -> bool:
        if action == self.pass_id:
            return self.last_act >= 0
        if action < 0 or action >= self.asp.n:
            return False
        hand = self.hands[self.turn]
        if (int(self.asp.mask[action]) & hand) != int(self.asp.mask[action]):
            return False
        if self.last_act < 0:
            return True
        t = int(self.asp.typ[action])
        if t == ROCKET:
            return int(self.asp.typ[self.last_act]) != ROCKET
        if t == BOMB:
            lt = int(self.asp.typ[self.last_act])
            if lt == ROCKET:
                return False
            if lt == BOMB:
                return int(self.asp.rank[action]) > int(self.asp.rank[self.last_act])
            return True
        if int(self.asp.typ[self.last_act]) != t:
            return False
        if int(self.asp.chain[action]) != int(self.asp.chain[self.last_act]):
            return False
        return int(self.asp.rank[action]) > int(self.asp.rank[self.last_act])

    # ---------- 推进 ----------
    def step(self, action: int) -> bool:
        if self.winner >= 0:
            raise RuntimeError("game already finished")
        if not self.is_legal(action):
            raise ValueError(f"illegal action {action} for player {self.turn}")
        p = self.turn
        if action == self.pass_id:
            self.passes += 1
            if self.passes >= 2:
                self.passes = 0
                self.last_act = -1
                self.turn = self.last_player
            else:
                self.turn = (p + 1) % 3
            self.history.append((p, action))
            return False
        mask = int(self.asp.mask[action])
        self.hands[p] = canonicalize_mask(self.hands[p] & ~mask)
        self.played[p] += self.asp.counts[action]
        self.passes = 0
        self.last_act = action
        self.last_player = p
        if self.hands[p] == 0:
            self.winner = p
            self.history.append((p, action))
            return True
        self.turn = (p + 1) % 3
        self.history.append((p, action))
        return False

    def rewards(self) -> list[float]:
        """地主 +1/-1，农民共享。"""
        if self.winner < 0:
            raise RuntimeError("game not finished")
        lw = 1.0 if self.winner == self.landlord else -1.0
        return [lw if p == self.landlord else -lw for p in range(3)]

    # ---------- 展示 ----------
    def hand_str(self, p: int) -> str:
        return mask_str(self.hands[p])

    def role(self, p: int) -> str:
        if p == self.landlord:
            return "地主"
        return "农民"


RANK_CHARS = "3 4 5 6 7 8 9 10 J Q K A 2".split() + ["小王", "大王"]


def mask_str(mask: int) -> str:
    cnt = mask_to_counts(mask)
    parts = []
    for r in range(N_RANKS):
        for _ in range(int(cnt[r])):
            parts.append(RANK_CHARS[r])
    return " ".join(parts)


def action_str(asp: ActionSpace, action: int) -> str:
    if action == asp.n:
        return "PASS"
    t = int(asp.typ[action])
    m = int(asp.mask[action])
    return f"[{t}] {mask_str(m)}"
