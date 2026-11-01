"""斗地主动作空间（无花色规范化）。

牌面用 54 bit 掩码表示：rank*4 + suit（3..2 为 rank 0..12），小王 bit52，大王 bit53。
动作按 rank 计数向量规范化，因此不含花色冗余。
"""
from __future__ import annotations

import numpy as np

N_RANKS = 15
N_NORMAL_RANKS = 13
FULL_DECK = (1 << 54) - 1

# 动作类型
SINGLE = 0
PAIR = 1
TRIO = 2
TRIO1 = 3
TRIO2 = 4
STRAIGHT = 5
PAIR_STRAIGHT = 6
PLANE = 7
PLANE1 = 8
PLANE2 = 9
FOUR2 = 10
FOUR2PAIR = 11
BOMB = 12
ROCKET = 13
N_TYPES = 14

TYPE_NAMES = [
    "单张", "对子", "三张", "三带一", "三带二", "顺子", "连对", "飞机",
    "飞机带单", "飞机带对", "四带二", "四带两对", "炸弹", "王炸",
]

PASS = -1


def counts_to_mask(counts) -> int:
    m = 0
    for r in range(N_RANKS):
        c = int(counts[r])
        if c <= 0:
            continue
        if r < N_NORMAL_RANKS:
            m |= ((1 << c) - 1) << (4 * r)
        else:
            m |= 1 << (52 + (r - N_NORMAL_RANKS))
    return m


def mask_to_counts(mask: int) -> np.ndarray:
    out = np.zeros(N_RANKS, dtype=np.int8)
    for r in range(N_NORMAL_RANKS):
        out[r] = bin((mask >> (4 * r)) & 0xF).count("1")
    if mask & (1 << 52):
        out[13] = 1
    if mask & (1 << 53):
        out[14] = 1
    return out


def _add(actions, counts, typ, rank, chain, ncards):
    actions.append((typ, rank, chain, ncards, counts_to_mask(counts)))


def generate_raw_actions():
    """生成全部合法动作（按 rank 计数规范化，飞机翅膀最多成对，避免牌型歧义）。"""
    A = []
    # 单张
    for r in range(N_RANKS):
        c = [0] * N_RANKS
        c[r] = 1
        _add(A, c, SINGLE, r, 0, 1)
    # 对子 / 三张 / 炸弹
    for r in range(N_NORMAL_RANKS):
        c = [0] * N_RANKS
        c[r] = 2
        _add(A, c, PAIR, r, 0, 2)
        c = [0] * N_RANKS
        c[r] = 3
        _add(A, c, TRIO, r, 0, 3)
        c = [0] * N_RANKS
        c[r] = 4
        _add(A, c, BOMB, r, 0, 4)
    # 三带一 / 三带二
    for t in range(N_NORMAL_RANKS):
        for k in range(N_RANKS):
            if k == t:
                continue
            c = [0] * N_RANKS
            c[t] = 3
            c[k] = 1
            _add(A, c, TRIO1, t, 0, 4)
        for k in range(N_NORMAL_RANKS):
            if k == t:
                continue
            c = [0] * N_RANKS
            c[t] = 3
            c[k] = 2
            _add(A, c, TRIO2, t, 0, 5)
    # 顺子
    for L in range(5, 13):
        for s in range(0, N_NORMAL_RANKS - L):
            c = [0] * N_RANKS
            for r in range(s, s + L):
                c[r] = 1
            _add(A, c, STRAIGHT, s, L, L)
    # 连对
    for L in range(3, 11):
        for s in range(0, N_NORMAL_RANKS - L):
            c = [0] * N_RANKS
            for r in range(s, s + L):
                c[r] = 2
            _add(A, c, PAIR_STRAIGHT, s, L, 2 * L)
    # 飞机（纯三顺）
    for L in range(2, 7):
        for s in range(0, N_NORMAL_RANKS - L):
            c = [0] * N_RANKS
            for r in range(s, s + L):
                c[r] = 3
            _add(A, c, PLANE, s, L, 3 * L)
    # 飞机带单（翅膀每个点数最多 2 张，避免与更长飞机歧义）
    for L in range(2, 6):
        for s in range(0, N_NORMAL_RANKS - L):
            chain = set(range(s, s + L))
            avail = [r for r in range(N_RANKS) if r not in chain]
            for wings in _multisets(avail, L, max_count=2):
                c = [0] * N_RANKS
                for r in range(s, s + L):
                    c[r] = 3
                for r in wings:
                    c[r] += 1
                _add(A, c, PLANE1, s, L, 4 * L)
    # 飞机带对
    for L in range(2, 5):
        for s in range(0, N_NORMAL_RANKS - L):
            chain = set(range(s, s + L))
            avail = [r for r in range(N_NORMAL_RANKS) if r not in chain]
            for wings in _multisets(avail, L, max_count=1):
                c = [0] * N_RANKS
                for r in range(s, s + L):
                    c[r] = 3
                for r in wings:
                    c[r] += 2
                _add(A, c, PLANE2, s, L, 5 * L)
    # 四带二（两张单牌，可以成对）
    for f in range(N_NORMAL_RANKS):
        avail = [r for r in range(N_RANKS) if r != f]
        for wings in _multisets(avail, 2, max_count=2):
            c = [0] * N_RANKS
            c[f] = 4
            for r in wings:
                c[r] += 1
            _add(A, c, FOUR2, f, 0, 6)
    # 四带两对
    for f in range(N_NORMAL_RANKS):
        avail = [r for r in range(N_NORMAL_RANKS) if r != f]
        for wings in _multisets(avail, 2, max_count=1):
            c = [0] * N_RANKS
            c[f] = 4
            for r in wings:
                c[r] = 2
            _add(A, c, FOUR2PAIR, f, 0, 8)
    # 王炸
    c = [0] * N_RANKS
    c[13] = c[14] = 1
    _add(A, c, ROCKET, 13, 0, 2)
    return A


def _multisets(items, k, max_count=4):
    """从 items 中取 k 张（可重复，每项最多 max_count），返回计数元组列表。"""
    items = list(items)
    out = []
    cur = []

    def rec(i, left):
        if left == 0:
            out.append(tuple(cur))
            return
        if i >= len(items):
            return
        # 跳过当前项
        rec(i + 1, left)
        # 使用 1..max_count 次；大小王各只有一张
        lim = 1 if items[i] >= 13 else max_count
        for cnt in range(1, min(lim, left) + 1):
            cur.extend([items[i]] * cnt)
            rec(i + 1, left - cnt)
            del cur[-cnt:]

    rec(0, k)
    return out


class ActionSpace:
    """排序后的动作表，附带按 (type, chain) 分组的索引。"""

    def __init__(self):
        raw = generate_raw_actions()
        # 去重（同一 mask 只保留一个，按生成顺序优先）
        seen = {}
        uniq = []
        for typ, rank, chain, ncards, mask in raw:
            if mask in seen:
                continue
            seen[mask] = True
            uniq.append((typ, rank, chain, ncards, mask))
        uniq.sort(key=lambda x: (x[0], x[2], x[1]))
        self.n = len(uniq)
        self.mask = np.array([u[4] for u in uniq], dtype=np.int64)
        self.typ = np.array([u[0] for u in uniq], dtype=np.int8)
        self.rank = np.array([u[1] for u in uniq], dtype=np.int8)
        self.chain = np.array([u[2] for u in uniq], dtype=np.int8)
        self.ncards = np.array([u[3] for u in uniq], dtype=np.int8)
        self.counts = np.zeros((self.n, N_RANKS), dtype=np.int8)
        for i, u in enumerate(uniq):
            m = int(u[4])
            for r in range(N_NORMAL_RANKS):
                self.counts[i, r] = bin((m >> (4 * r)) & 0xF).count("1")
            if m & (1 << 52):
                self.counts[i, 13] = 1
            if m & (1 << 53):
                self.counts[i, 14] = 1
        # 分组 key = typ*16 + chain
        self.group_start = np.full(256, -1, dtype=np.int32)
        self.group_end = np.full(256, -1, dtype=np.int32)
        keys = self.typ.astype(np.int32) * 16 + self.chain.astype(np.int32)
        i = 0
        while i < self.n:
            j = i
            while j + 1 < self.n and keys[j + 1] == keys[i]:
                j += 1
            self.group_start[keys[i]] = i
            self.group_end[keys[i]] = j + 1
            i = j + 1
        # 预计算动作特征
        self.features = self._build_features()

    def _build_features(self):
        n = self.n
        feat = np.zeros((n, N_RANKS + N_TYPES + 5), dtype=np.float32)
        feat[:, :N_RANKS] = self.counts.astype(np.float32) / 4.0
        for t in range(N_TYPES):
            feat[:, N_RANKS + t] = (self.typ == t).astype(np.float32)
        feat[:, N_RANKS + N_TYPES + 0] = self.rank.astype(np.float32) / 14.0
        feat[:, N_RANKS + N_TYPES + 1] = self.chain.astype(np.float32) / 12.0
        feat[:, N_RANKS + N_TYPES + 2] = self.ncards.astype(np.float32) / 20.0
        feat[:, N_RANKS + N_TYPES + 3] = (self.typ == BOMB).astype(np.float32)
        feat[:, N_RANKS + N_TYPES + 4] = (self.typ == ROCKET).astype(np.float32)
        return feat


ACTION_FEAT_DIM = N_RANKS + N_TYPES + 5
