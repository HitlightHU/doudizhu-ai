"""Numba 批量斗地主引擎：发牌、合法动作枚举、状态推进、特征提取。

所有函数都是无花色的 rank 级表示；一局游戏的状态放在固定大小数组中，
批量接口一次处理 G 局，供自对弈数据生成使用。
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .actions import (
    BOMB, FULL_DECK, N_RANKS, N_TYPES, ROCKET,
)

MAXA = 2048         # 单个决策点最大合法动作数（超出则截断，极罕见）
HIST = 15           # 历史动作窗口
STATE_DIM = 132
HIST_DIM = 33
N_STATE_VECS = 7

POPCOUNT_OK = True


# ----------------------------------------------------------------------------
# 基础工具（Numba）
# ----------------------------------------------------------------------------
@njit(inline="always")
def _pop4(x):
    return ((x) & 1) + ((x >> 1) & 1) + ((x >> 2) & 1) + ((x >> 3) & 1)


@njit(inline="always")
def _count_cards(mask):
    c = 0
    for r in range(13):
        c += _pop4((mask >> (4 * r)) & 15)
    c += (mask >> 52) & 1
    c += (mask >> 53) & 1
    return c


@njit(inline="always")
def _canon_mask(m):
    out = np.int64(0)
    for r in range(13):
        c = _pop4((m >> (4 * r)) & 15)
        out |= ((np.int64(1) << c) - 1) << (4 * r)
    if (m >> 52) & 1:
        out |= np.int64(1) << 52
    if (m >> 53) & 1:
        out |= np.int64(1) << 53
    return out


@njit(inline="always")
def _write_cnt_row(out, base, cnt):
    for r in range(15):
        out[base + r] = cnt[r] * 0.25


@njit(inline="always")
def _write_counts(out, base, mask):
    for r in range(13):
        out[base + r] = ((mask >> (4 * r)) & 15) * 0.25
    out[base + 13] = ((mask >> 52) & 1) * 0.25
    out[base + 14] = ((mask >> 53) & 1) * 0.25


@njit(inline="always")
def _next_int(state, n):
    # xorshift64*，state 为长度 1 的 int64 数组
    x = state[0]
    x ^= (x >> 12)
    x ^= (x << 25) & 0x7FFFFFFFFFFFFFFF
    x ^= (x >> 27)
    state[0] = x
    v = (x * 2685821657736338717) & 0x7FFFFFFFFFFFFFFF
    return v % n


@njit(inline="always")
def _mask_from_cnt1(cnt):
    m = np.int64(0)
    for r in range(13):
        c = cnt[r]
        m |= ((np.int64(1) << c) - 1) << (4 * r)
    if cnt[13] > 0:
        m |= np.int64(1) << 52
    if cnt[14] > 0:
        m |= np.int64(1) << 53
    return m


@njit(inline="always")
def _mask_from_cnt2(cnt, p):
    m = np.int64(0)
    for r in range(13):
        c = cnt[p, r]
        m |= ((np.int64(1) << c) - 1) << (4 * r)
    if cnt[p, 13] > 0:
        m |= np.int64(1) << 52
    if cnt[p, 14] > 0:
        m |= np.int64(1) << 53
    return m


@njit(cache=True)
def _deal_one(g, hands, played, bottom, landlord, turn, last_act, last_player,
              passes, winner, hist_a, hist_p, hist_len, rng, deck, cnt, bcnt):
    for i in range(52):
        deck[i] = i // 4
    deck[52] = 13
    deck[53] = 14
    for i in range(53, 0, -1):
        j = _next_int(rng, i + 1)
        tmp = deck[i]
        deck[i] = deck[j]
        deck[j] = tmp
    for r in range(15):
        bcnt[r] = 0
        for p in range(3):
            cnt[p, r] = 0
    for i in range(51):
        cnt[i % 3, deck[i]] += 1
    for i in range(51, 54):
        bcnt[deck[i]] += 1
    hands[g, 0] = _mask_from_cnt2(cnt, 0)
    hands[g, 1] = _mask_from_cnt2(cnt, 1)
    hands[g, 2] = _mask_from_cnt2(cnt, 2)
    b = _mask_from_cnt1(bcnt)
    for r in range(15):
        played[g, 0, r] = 0
        played[g, 1, r] = 0
        played[g, 2, r] = 0
    bottom[g] = b
    landlord[g] = -1
    turn[g] = -1
    last_act[g] = -1
    last_player[g] = -1
    passes[g] = 0
    winner[g] = -1
    for i in range(HIST):
        hist_a[g, i] = -1
        hist_p[g, i] = -1
    hist_len[g] = 0


@njit(cache=True)
def reset_all(hands, played, bottom, landlord, turn, last_act, last_player, passes,
              winner, hist_a, hist_p, hist_len, rng):
    G = hands.shape[0]
    deck = np.empty(54, dtype=np.int64)
    cnt = np.zeros((3, 15), dtype=np.int8)
    bcnt = np.zeros(15, dtype=np.int8)
    for g in range(G):
        _deal_one(g, hands, played, bottom, landlord, turn, last_act, last_player,
                  passes, winner, hist_a, hist_p, hist_len, rng, deck, cnt, bcnt)


@njit(cache=True)
def reset_slots(slots, hands, played, bottom, landlord, turn, last_act, last_player,
                passes, winner, hist_a, hist_p, hist_len, rng):
    deck = np.empty(54, dtype=np.int64)
    cnt = np.zeros((3, 15), dtype=np.int8)
    bcnt = np.zeros(15, dtype=np.int8)
    for k in range(slots.shape[0]):
        g = slots[k]
        _deal_one(g, hands, played, bottom, landlord, turn, last_act, last_player,
                  passes, winner, hist_a, hist_p, hist_len, rng, deck, cnt, bcnt)


@njit(cache=True)
def assign_landlords(slots, landlords, hands, bottom, landlord, turn, last_act,
                     last_player, passes, winner, hist_a, hist_p, hist_len):
    """把底牌并入指定地主手牌，并重置出牌阶段状态。"""
    for k in range(slots.shape[0]):
        g = slots[k]
        l = landlords[k]
        h = hands[g, l]
        b = bottom[g]
        out = np.int64(0)
        for r in range(13):
            c = _pop4((h >> (4 * r)) & 15) + _pop4((b >> (4 * r)) & 15)
            out |= ((np.int64(1) << c) - 1) << (4 * r)
        if ((h | b) >> 52) & 1:
            out |= np.int64(1) << 52
        if ((h | b) >> 53) & 1:
            out |= np.int64(1) << 53
        hands[g, l] = out
        landlord[g] = l
        turn[g] = l
        last_act[g] = -1
        last_player[g] = -1
        passes[g] = 0
        winner[g] = -1
        hist_len[g] = 0
        for i in range(HIST):
            hist_a[g, i] = -1
            hist_p[g, i] = -1


# ----------------------------------------------------------------------------
# 合法动作
# ----------------------------------------------------------------------------
@njit(inline="always")
def _push(out, g, n, i):
    if n < MAXA:
        out[g, n] = i
    return n + 1


@njit(cache=True)
def legal_batch(hands, turn, last_act, act_mask, act_type, act_rank, act_chain,
                group_start, group_end, pass_idx, out_idx, out_cnt):
    G = hands.shape[0]
    N = act_mask.shape[0]
    for g in range(G):
        hand = hands[g, turn[g]]
        n = 0
        la = last_act[g]
        if la < 0:
            for i in range(N):
                m = np.int64(act_mask[i])
                if (m & hand) == m:
                    n = _push(out_idx, g, n, i)
        else:
            t = act_type[la]
            c = act_chain[la]
            r = act_rank[la]
            if t != ROCKET:
                if t == BOMB:
                    key = BOMB * 16
                    s = group_start[key]
                    e = group_end[key]
                    lo = s
                    hi = e
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if act_rank[mid] <= r:
                            lo = mid + 1
                        else:
                            hi = mid
                    for i in range(lo, e):
                        m = np.int64(act_mask[i])
                        if (m & hand) == m:
                            n = _push(out_idx, g, n, i)
                else:
                    key = t * 16 + c
                    s = group_start[key]
                    e = group_end[key]
                    lo = s
                    hi = e
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if act_rank[mid] <= r:
                            lo = mid + 1
                        else:
                            hi = mid
                    for i in range(lo, e):
                        m = np.int64(act_mask[i])
                        if (m & hand) == m:
                            n = _push(out_idx, g, n, i)
                    # 炸弹
                    key = BOMB * 16
                    bs = group_start[key]
                    be = group_end[key]
                    for i in range(bs, be):
                        m = np.int64(act_mask[i])
                        if (m & hand) == m:
                            n = _push(out_idx, g, n, i)
                # 王炸
                key = ROCKET * 16
                rid = group_start[key]
                m = np.int64(act_mask[rid])
                if (m & hand) == m:
                    n = _push(out_idx, g, n, rid)
            n = _push(out_idx, g, n, pass_idx)
        if n > MAXA:
            n = MAXA
        out_cnt[g] = n


# ----------------------------------------------------------------------------
# 推进
# ----------------------------------------------------------------------------
@njit(cache=True)
def apply_batch(hands, played, turn, last_act, last_player, passes, winner,
                hist_a, hist_p, hist_len, actions, pass_idx, act_mask, act_counts):
    G = hands.shape[0]
    for g in range(G):
        if winner[g] >= 0:
            continue
        p = turn[g]
        a = actions[g]
        hval = a
        if a >= 0 and a != pass_idx:
            m = act_mask[a]
            hands[g, p] = _canon_mask(hands[g, p] & ~m)
            for r in range(15):
                played[g, p, r] += act_counts[a, r]
            last_act[g] = a
            last_player[g] = p
            passes[g] = 0
            if hands[g, p] == 0:
                winner[g] = p
            else:
                turn[g] = (p + 1) % 3
        else:
            hval = -1
            passes[g] += 1
            if passes[g] >= 2:
                passes[g] = 0
                last_act[g] = -1
                turn[g] = last_player[g]
            else:
                turn[g] = (p + 1) % 3
        hl = hist_len[g]
        if hl < HIST:
            hist_a[g, hl] = hval
            hist_p[g, hl] = p
            hist_len[g] = hl + 1
        else:
            for i in range(1, HIST):
                hist_a[g, i - 1] = hist_a[g, i]
                hist_p[g, i - 1] = hist_p[g, i]
            hist_a[g, HIST - 1] = hval
            hist_p[g, HIST - 1] = p


# ----------------------------------------------------------------------------
# 特征
# ----------------------------------------------------------------------------
@njit(cache=True)
def observe_batch(hands, played, bottom, landlord, turn, last_act, last_player, passes,
                  hist_a, hist_p, hist_len, act_counts, act_type,
                  out_state, out_hist):
    G = hands.shape[0]
    for g in range(G):
        p = turn[g]
        hand = hands[g, p]
        _write_counts(out_state[g], 0, hand)
        _write_cnt_row(out_state[g], 15, played[g, p])
        _write_cnt_row(out_state[g], 30, played[g, (p + 1) % 3])
        _write_cnt_row(out_state[g], 45, played[g, (p + 2) % 3])
        unseen = np.int64(0)
        is_farmer = p != landlord[g]
        for r in range(13):
            own_c = _pop4((hand >> (4 * r)) & 15)
            pl_c = played[g, 0, r] + played[g, 1, r] + played[g, 2, r]
            bt_c = _pop4((bottom[g] >> (4 * r)) & 15) if is_farmer else 0
            rem = 4 - own_c - pl_c - bt_c
            if rem < 0:
                rem = 0
            unseen |= ((np.int64(1) << rem) - 1) << (4 * r)
        own_j = (hand >> 52) & 1
        pl_j = played[g, 0, 13] + played[g, 1, 13] + played[g, 2, 13]
        bt_j = ((bottom[g] >> 52) & 1) if is_farmer else 0
        if 1 - own_j - pl_j - bt_j > 0:
            unseen |= np.int64(1) << 52
        own_j = (hand >> 53) & 1
        pl_j = played[g, 0, 14] + played[g, 1, 14] + played[g, 2, 14]
        bt_j = ((bottom[g] >> 53) & 1) if is_farmer else 0
        if 1 - own_j - pl_j - bt_j > 0:
            unseen |= np.int64(1) << 53
        _write_counts(out_state[g], 60, unseen)
        la = last_act[g]
        if la >= 0:
            for r in range(15):
                out_state[g, 75 + r] = act_counts[la, r] * 0.25
        else:
            for r in range(15):
                out_state[g, 75 + r] = 0.0
        _write_counts(out_state[g], 90, bottom[g])
        # 标量
        own_c = _count_cards(hand)
        nc = _count_cards(hands[g, (p + 1) % 3])
        pc = _count_cards(hands[g, (p + 2) % 3])
        out_state[g, 105] = own_c / 20.0
        out_state[g, 106] = nc / 20.0
        out_state[g, 107] = pc / 20.0
        for k in range(3):
            out_state[g, 108 + k] = 0.0
        if p == landlord[g]:
            out_state[g, 108] = 1.0
        elif p == (landlord[g] + 1) % 3:
            out_state[g, 109] = 1.0
        else:
            out_state[g, 110] = 1.0
        for k in range(15):
            out_state[g, 111 + k] = 0.0
        if la >= 0:
            out_state[g, 111 + act_type[la]] = 1.0
        else:
            out_state[g, 111 + 14] = 1.0
        for k in range(4):
            out_state[g, 126 + k] = 0.0
        if la >= 0:
            rel = (last_player[g] - p + 3) % 3
            out_state[g, 126 + rel] = 1.0
        else:
            out_state[g, 129] = 1.0
        out_state[g, 130] = 1.0 if la < 0 else 0.0
        out_state[g, 131] = passes[g] * 0.5
        # 历史窗口（右对齐，时间从旧到新）
        hl = hist_len[g]
        start = HIST - hl
        for i in range(HIST):
            base = i * HIST_DIM
            for k in range(HIST_DIM):
                out_hist[g, base + k] = 0.0
            if i >= start:
                a = hist_a[g, i - start]
                pl = hist_p[g, i - start]
                if a >= 0:
                    for r in range(15):
                        out_hist[g, base + r] = act_counts[a, r] * 0.25
                    out_hist[g, base + 15 + act_type[a]] = 1.0
                else:
                    out_hist[g, base + 15 + 14] = 1.0
                rel = (pl - p + 3) % 3
                out_hist[g, base + 30 + rel] = 1.0


class BatchEngine:
    """批量游戏状态容器。"""

    def __init__(self, n_games: int, seed: int = 0, action_space=None,
                 auto_landlord: bool = True):
        from .reference import get_action_space
        self.asp = action_space or get_action_space()
        self.auto_landlord = bool(auto_landlord)
        self.G = int(n_games)
        G = self.G
        self.hands = np.zeros((G, 3), dtype=np.int64)
        self.played = np.zeros((G, 3, 15), dtype=np.int8)
        self.bottom = np.zeros(G, dtype=np.int64)
        self.landlord = np.zeros(G, dtype=np.int8)
        self.turn = np.zeros(G, dtype=np.int8)
        self.last_act = np.full(G, -1, dtype=np.int32)
        self.last_player = np.full(G, -1, dtype=np.int8)
        self.passes = np.zeros(G, dtype=np.int8)
        self.winner = np.full(G, -1, dtype=np.int8)
        self.hist_a = np.full((G, HIST), -1, dtype=np.int32)
        self.hist_p = np.full((G, HIST), -1, dtype=np.int8)
        self.hist_len = np.zeros(G, dtype=np.int8)
        self.rng = np.array([seed if seed != 0 else 88172645463325252], dtype=np.int64)
        self.np_rng = np.random.default_rng(seed + 17 if seed else 12345)
        # 动作表 + PASS 行
        n = self.asp.n
        self.act_counts = np.zeros((n + 1, N_RANKS), dtype=np.int8)
        self.act_counts[:n] = self.asp.counts
        self.pass_idx = n
        self.legal_idx = np.zeros((G, MAXA), dtype=np.int32)
        self.legal_cnt = np.zeros(G, dtype=np.int32)
        self.state = np.zeros((G, STATE_DIM), dtype=np.float32)
        self.hist = np.zeros((G, HIST * HIST_DIM), dtype=np.float32)
        self._cnt = np.zeros((3, 15), dtype=np.int8)
        self._bcnt = np.zeros(15, dtype=np.int8)
        self._reset_all()

    def _reset_all(self):
        reset_all(self.hands, self.played, self.bottom, self.landlord, self.turn,
                  self.last_act, self.last_player, self.passes, self.winner,
                  self.hist_a, self.hist_p, self.hist_len, self.rng)
        if self.auto_landlord:
            self.assign_landlords(np.arange(self.G, dtype=np.int32),
                                  self.np_rng.integers(0, 3, size=self.G).astype(np.int8))

    def reset_slots(self, slots):
        slots = np.asarray(slots, dtype=np.int32)
        if slots.size == 0:
            return
        reset_slots(slots, self.hands, self.played, self.bottom, self.landlord, self.turn,
                    self.last_act, self.last_player, self.passes, self.winner,
                    self.hist_a, self.hist_p, self.hist_len, self.rng)
        if self.auto_landlord:
            self.assign_landlords(slots, self.np_rng.integers(0, 3, size=len(slots)).astype(np.int8))

    def assign_landlords(self, slots, landlords):
        slots = np.asarray(slots, dtype=np.int32)
        landlords = np.asarray(landlords, dtype=np.int8)
        assign_landlords(slots, landlords, self.hands, self.bottom, self.landlord, self.turn,
                         self.last_act, self.last_player, self.passes, self.winner,
                         self.hist_a, self.hist_p, self.hist_len)

    def set_states(self, hands, played, bottom, landlord, turn, last_act, last_player,
                   passes, hist_a, hist_p, hist_len):
        """用于自定义开局/中局状态（分析模式、rollout）。"""
        self.hands[:] = np.asarray(hands, dtype=np.int64)
        self.played[:] = np.asarray(played, dtype=np.int8)
        self.bottom[:] = np.asarray(bottom, dtype=np.int64)
        self.landlord[:] = np.asarray(landlord, dtype=np.int8)
        self.turn[:] = np.asarray(turn, dtype=np.int8)
        self.last_act[:] = np.asarray(last_act, dtype=np.int32)
        self.last_player[:] = np.asarray(last_player, dtype=np.int8)
        self.passes[:] = np.asarray(passes, dtype=np.int8)
        self.hist_a[:] = np.asarray(hist_a, dtype=np.int32)
        self.hist_p[:] = np.asarray(hist_p, dtype=np.int8)
        self.hist_len[:] = np.asarray(hist_len, dtype=np.int8)
        self.winner[:] = -1

    def observe(self):
        observe_batch(self.hands, self.played, self.bottom, self.landlord, self.turn,
                      self.last_act, self.last_player, self.passes,
                      self.hist_a, self.hist_p, self.hist_len,
                      self.act_counts, self.asp.typ,
                      self.state, self.hist)
        return self.state, self.hist

    def legal(self):
        legal_batch(self.hands, self.turn, self.last_act, self.asp.mask, self.asp.typ,
                    self.asp.rank, self.asp.chain, self.asp.group_start, self.asp.group_end,
                    self.pass_idx, self.legal_idx, self.legal_cnt)
        return self.legal_idx, self.legal_cnt

    def step(self, actions):
        apply_batch(self.hands, self.played, self.turn, self.last_act, self.last_player,
                    self.passes, self.winner, self.hist_a, self.hist_p, self.hist_len,
                    np.asarray(actions, dtype=np.int32), self.pass_idx, self.asp.mask,
                    self.act_counts)

    def finished(self):
        return self.winner >= 0
