"""Determinization + 完整对局 rollout 工具。

用于：
- 生成叫地主训练数据（给定 17 张手牌，模拟当地主/当农民的胜率）
- 分析模式：根据已知信息采样对手手牌，评估当前局面的胜率与候选动作
"""
from __future__ import annotations

import numpy as np
import torch

from .actions import N_RANKS, counts_to_mask, mask_to_counts
from .bidnet import TOTAL_COUNTS, hand_counts
from .engine import HIST, HIST_DIM
from .selfplay import select_actions_batch


def _counts_add(*arrs):
    out = np.zeros(N_RANKS, dtype=np.int16)
    for a in arrs:
        out += np.asarray(a, dtype=np.int16)
    return out.astype(np.int8)


def sample_pool_split(pool_counts: np.ndarray, sizes, rng) -> list[np.ndarray]:
    ids = np.repeat(np.arange(N_RANKS, dtype=np.int64),
                    np.asarray(pool_counts, dtype=np.int64))
    rng.shuffle(ids)
    out = []
    pos = 0
    for s in sizes:
        cnt = np.zeros(N_RANKS, dtype=np.int8)
        for r in ids[pos:pos + s]:
            cnt[int(r)] += 1
        out.append(cnt)
        pos += int(s)
    return out


def _counts_to_mask_list(cnt_rows) -> np.ndarray:
    return np.array([counts_to_mask(c) for c in cnt_rows], dtype=np.int64)


def bid_states(hand_masks: np.ndarray, role: int, samples: int, rng):
    """给定一批 17 张手牌，构造 role=0 当地主 / role=1 当农民的 rollout 初始状态。

    返回 dict：hands [G,3], bottom, landlord, human_seat
    """
    hand_masks = np.asarray(hand_masks, dtype=np.int64).reshape(-1)
    N = len(hand_masks)
    G = N * samples
    hands = np.zeros((G, 3), dtype=np.int64)
    bottom = np.zeros(G, dtype=np.int64)
    landlord = np.zeros(G, dtype=np.int8)
    human_seat = np.zeros(G, dtype=np.int8)
    g = 0
    for i in range(N):
        hcnt = mask_to_counts(int(hand_masks[i]))
        rem = TOTAL_COUNTS - hcnt
        for s in range(samples):
            opp1, opp2, bcnt = sample_pool_split(rem, [17, 17, 3], rng)
            bm = counts_to_mask(bcnt)
            bottom[g] = bm
            if role == 0:
                hands[g, 0] = counts_to_mask(_counts_add(hcnt, bcnt))
                hands[g, 1] = counts_to_mask(opp1)
                hands[g, 2] = counts_to_mask(opp2)
                landlord[g] = 0
            else:
                seat = 1 if ((i + s) % 2 == 0) else 2
                hands[g, 0] = int(hand_masks[i])
                hands[g, seat] = counts_to_mask(_counts_add(opp1 if seat == 1 else opp2, bcnt))
                hands[g, 3 - seat] = counts_to_mask(opp2 if seat == 1 else opp1)
                landlord[g] = seat
            human_seat[g] = 0
            g += 1
    played = np.zeros((G, 3, N_RANKS), dtype=np.int8)
    return {"hands": hands, "played": played, "bottom": bottom, "landlord": landlord,
            "turn": landlord.copy(), "last_act": np.full(G, -1, dtype=np.int32),
            "last_player": np.full(G, -1, dtype=np.int8),
            "passes": np.zeros(G, dtype=np.int8), "human_seat": human_seat,
            "hist_a": np.full((G, HIST), -1, dtype=np.int32),
            "hist_p": np.full((G, HIST), -1, dtype=np.int8),
            "hist_len": np.zeros(G, dtype=np.int8)}


def known_states(human_hand: int, human_seat: int, played: np.ndarray, bottom: int,
                 landlord: int, turn: int, last_act: int, last_player: int, passes: int,
                 history, hand_sizes, samples: int, rng):
    """从已知信息采样对手手牌，构造 samples 个一致的 rollout 初始状态。"""
    human_seat = int(human_seat)
    landlord = int(landlord)
    played = np.asarray(played, dtype=np.int8)
    # 对手手牌并集 = 全牌 - 自己手牌 - 已出牌（底牌仍在对手手里，不额外扣除）
    unseen = TOTAL_COUNTS.astype(np.int16) - mask_to_counts(int(human_hand)).astype(np.int16) - played.sum(0).astype(np.int16)
    unseen = np.maximum(unseen, 0)
    opp_seats = [p for p in range(3) if p != human_seat]
    sizes = [int(hand_sizes[p]) for p in opp_seats]
    diff = int(unseen.sum()) - sum(sizes)
    if diff != 0:
        sizes[-1] += diff
        if sizes[-1] < 0:
            sizes[-1] = 0

    hands = np.zeros((samples, 3), dtype=np.int64)
    for s in range(samples):
        parts = sample_pool_split(unseen, sizes, rng)
        hands[s, human_seat] = int(human_hand)
        for k, p in enumerate(opp_seats):
            hands[s, p] = counts_to_mask(parts[k])

    played_b = np.tile(played[None, :, :], (samples, 1, 1))
    bottom_b = np.full(samples, int(bottom), dtype=np.int64)
    landlord_b = np.full(samples, landlord, dtype=np.int8)
    turn_b = np.full(samples, int(turn), dtype=np.int8)
    last_b = np.full(samples, int(last_act), dtype=np.int32)
    lastp_b = np.full(samples, int(last_player), dtype=np.int8)
    passes_b = np.full(samples, int(passes), dtype=np.int8)

    hist = list(history)[-HIST:]
    hist_a = np.full((samples, HIST), -1, dtype=np.int32)
    hist_p = np.full((samples, HIST), -1, dtype=np.int8)
    hist_len = np.full(samples, len(hist), dtype=np.int8)
    for i, (p, a) in enumerate(hist):
        hist_a[:, i] = int(a)
        hist_p[:, i] = int(p)
    return {"hands": hands, "played": played_b, "bottom": bottom_b, "landlord": landlord_b,
            "turn": turn_b, "last_act": last_b, "last_player": lastp_b, "passes": passes_b,
            "hist_a": hist_a, "hist_p": hist_p, "hist_len": hist_len}


def states_to_engine(engine, st):
    engine.set_states(st["hands"], st["played"], st["bottom"], st["landlord"], st["turn"],
                      st["last_act"], st["last_player"], st["passes"],
                      st["hist_a"], st["hist_p"], st["hist_len"])


def run_policy(engine, model, device, action_feat, max_steps: int = 400):
    """所有座位都用模型策略打完整局，返回 winner [G]。"""
    G = engine.G
    winners = np.full(G, -1, dtype=np.int8)
    active = np.arange(G)
    acts = np.zeros(G, dtype=np.int32)
    steps = 0
    while active.size and steps < max_steps:
        st, hist = engine.observe()
        idx, cnt = engine.legal()
        a = select_actions_batch(model, device, action_feat, st[active], hist[active],
                                 idx[active], cnt[active], 0.0)
        acts[active] = a
        engine.step(acts)
        w = engine.winner
        newly = active[w[active] >= 0]
        winners[newly] = w[newly]
        active = active[w[active] < 0]
        steps += 1
    return winners


def win_flags(winners: np.ndarray, landlord: np.ndarray, human_seat: int) -> np.ndarray:
    is_ll = human_seat == landlord
    return np.where(is_ll, winners == human_seat, winners != landlord)
