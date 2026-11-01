"""基准机器人：随机、贪心、启发式。只使用公开信息。"""
from __future__ import annotations

import numpy as np

from .actions import BOMB, ROCKET, TRIO, TRIO1, TRIO2, PLANE, PLANE1, PLANE2, mask_to_counts
from .reference import get_action_space


class RandomBot:
    name = "random"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def choose(self, hand, last_act, legal, turn, landlord, last_player, sizes):
        return int(self.rng.choice(legal))


class GreedyBot:
    """能压就压最小的，队友出牌就过。"""
    name = "greedy"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def choose(self, hand, last_act, legal, turn, landlord, last_player, sizes):
        asp = get_action_space()
        pass_id = asp.n
        acts = [a for a in legal if a != pass_id]
        if not acts:
            return pass_id
        if last_act >= 0 and last_player != landlord and turn != landlord:
            return pass_id
        acts.sort(key=lambda a: (int(asp.rank[a]), int(asp.ncards[a])))
        return int(acts[0])


class HeuristicBot:
    """手工规则：优先走小牌、保留炸弹、农民配合、残局压制。"""
    name = "heuristic"

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def choose(self, hand, last_act, legal, turn, landlord, last_player, sizes):
        asp = get_action_space()
        pass_id = asp.n
        legal = np.asarray(legal, dtype=np.int64)
        acts = legal[legal != pass_id]
        if acts.size == 0:
            return pass_id
        hand_cnt = mask_to_counts(hand).astype(np.int32)
        hand_n = int(hand_cnt.sum())
        ncards = asp.ncards[acts].astype(np.int32)
        types = asp.typ[acts].astype(np.int32)
        ranks = asp.rank[acts].astype(np.int32)
        cnts = asp.counts[acts].astype(np.int32)  # [k,15]

        # 1) 一把走完
        win = np.nonzero(ncards == hand_n)[0]
        if win.size:
            return int(acts[win[np.argmin(ncards[win])]])

        is_landlord = turn == landlord
        partner = None
        if not is_landlord:
            partner = 3 - turn - landlord
        opp_cards = sizes[landlord] if not is_landlord else min(sizes[(landlord + 1) % 3], sizes[(landlord + 2) % 3])
        partner_cards = sizes[partner] if partner is not None else 99

        following = last_act >= 0
        last_is_partner = following and partner is not None and last_player == partner

        # 队友出的牌：一般不压
        if last_is_partner:
            if partner_cards <= 2:
                return pass_id
            # 队友出的是小牌且地主快走完，必要时接管
            if not (opp_cards <= 3 and int(asp.rank[last_act]) <= 8):
                return pass_id

        # 计算代价
        high = cnts[:, 12] * 1.2 + cnts[:, 13] * 3.0 + cnts[:, 14] * 4.0
        brk = np.zeros(acts.size, dtype=np.float32)
        for r in range(13):
            c = hand_cnt[r]
            used = cnts[:, r]
            if c == 4:
                brk += np.where((used > 0) & (used < 4), 9.0, 0.0)
            elif c == 3:
                brk += np.where(used == 1, 1.2, 0.0)
            elif c == 2:
                brk += np.where(used == 1, 0.5, 0.0)
        bomb = ((types == BOMB).astype(np.float32) * 5.0 +
                (types == ROCKET).astype(np.float32) * 9.0)

        if following:
            cost = (ranks * 1.0 - ncards * 0.15 + high * 1.0 + brk * 1.0 + bomb)
            # 只剩炸弹可压：对手牌少才用
            if np.all(bomb > 0) and opp_cards > 3 and hand_n > 6:
                return pass_id
            best = int(np.argmin(cost))
            if cost[best] > 12.0 and opp_cards > 4 and hand_n > 8:
                return pass_id
            return int(acts[best])
        else:
            # 出牌：优先长低牌，避免拆炸弹；对手只剩 1 张时避免送单张
            cost = (ranks * 1.2 - ncards * 0.85 + high * 1.1 + brk * 1.2 + bomb * 1.5)
            if opp_cards == 1:
                # 尽量出对子/多张，少出单张
                cost = cost + np.where(ncards == 1, 3.0, 0.0)
            best = int(np.argmin(cost))
            return int(acts[best])

    def choose_from_game(self, g):
        legal = g.legal_actions()
        sizes = [g.card_count(p) for p in range(3)]
        return self.choose(g.hands[g.turn], g.last_act, legal, g.turn, g.landlord,
                           g.last_player, sizes)


BOT_REGISTRY = {
    "random": RandomBot,
    "greedy": GreedyBot,
    "heuristic": HeuristicBot,
}


def make_bot(name: str, seed=None):
    return BOT_REGISTRY[name](seed)
