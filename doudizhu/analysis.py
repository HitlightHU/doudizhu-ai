"""分析模式 / GUI 的已知信息状态与 Monte-Carlo 推理。

已知：自己的手牌、三家已出牌、底牌、地主、剩余张数、上一手、历史。
未知：对手手牌——按剩余牌随机采样（determinization），再用当前模型打完整局。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .actions import N_RANKS, ROCKET, BOMB, counts_to_mask, mask_to_counts
from .bidnet import TOTAL_COUNTS, score_hands
from .engine import HIST, HIST_DIM, BatchEngine
from .reference import canonicalize_mask, get_action_space
from .rollout import (bid_states, known_states, run_policy, sample_pool_split,
                      states_to_engine, win_flags)


@dataclass
class AnalysisSession:
    human_seat: int = 0
    landlord: int = 0
    bottom: int = 0
    human_hand: int = 0
    hand_sizes: list = field(default_factory=lambda: [17, 17, 17])
    played: np.ndarray = field(default_factory=lambda: np.zeros((3, N_RANKS), dtype=np.int8))
    turn: int = 0
    last_act: int = -1
    last_player: int = -1
    passes: int = 0
    history: list = field(default_factory=list)
    winner: int = -1
    asp: object = None

    def __post_init__(self):
        if self.asp is None:
            self.asp = get_action_space()

    # ---------- 初始化 ----------
    @classmethod
    def start(cls, human_hand: int, human_is_landlord: bool, bottom: int, landlord_seat: int):
        s = cls()
        s.human_seat = 0
        s.bottom = int(bottom)
        if human_is_landlord:
            s.landlord = 0
            s.human_hand = counts_to_mask(mask_to_counts(int(human_hand)) + mask_to_counts(int(bottom)))
            s.hand_sizes = [20, 17, 17]
        else:
            s.landlord = int(landlord_seat)
            s.human_hand = int(human_hand)
            s.hand_sizes = [17, 17, 17]
            s.hand_sizes[s.landlord] = 20
        s.turn = s.landlord
        s.last_act = -1
        s.last_player = -1
        s.passes = 0
        s.played = np.zeros((3, N_RANKS), dtype=np.int8)
        s.history = []
        s.winner = -1
        return s

    # ---------- 查询 ----------
    def human_is_landlord(self) -> bool:
        return self.human_seat == self.landlord

    def current_player(self) -> int:
        return self.turn

    def legal_actions(self, hand: int | None = None):
        """返回当前出牌人（默认自己）的合法动作；调用方保证轮到自己时使用。"""
        asp = self.asp
        hand = self.human_hand if hand is None else hand
        out = []
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
            return [asp.n]
        if t == BOMB:
            s = int(asp.group_start[BOMB * 16])
            e = int(asp.group_end[BOMB * 16])
            for i in range(s, e):
                if int(asp.rank[i]) > r and (int(asp.mask[i]) & hand) == int(asp.mask[i]):
                    out.append(i)
            rid = int(asp.group_start[ROCKET * 16])
            if (int(asp.mask[rid]) & hand) == int(asp.mask[rid]):
                out.append(rid)
            out.append(asp.n)
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
            if (int(asp.mask[i]) & hand) == int(asp.mask[i]):
                out.append(i)
        rid = int(asp.group_start[ROCKET * 16])
        if (int(asp.mask[rid]) & hand) == int(asp.mask[rid]):
            out.append(rid)
        out.append(asp.n)
        return out

    def is_legal_for(self, player: int, action: int) -> bool:
        asp = self.asp
        if self.winner >= 0:
            return False
        if action == asp.n:
            return self.last_act >= 0
        if action < 0 or action >= asp.n:
            return False
        if int(asp.ncards[action]) > self.hand_sizes[player]:
            return False
        if player == self.human_seat and (int(asp.mask[action]) & self.human_hand) != int(asp.mask[action]):
            return False
        if self.last_act < 0:
            return True
        la = self.last_act
        lt = int(asp.typ[la])
        t = int(asp.typ[action])
        if t == ROCKET:
            return lt != ROCKET
        if t == BOMB:
            if lt == ROCKET:
                return False
            if lt == BOMB:
                return int(asp.rank[action]) > int(asp.rank[la])
            return True
        if lt != t:
            return False
        if int(asp.chain[action]) != int(asp.chain[la]):
            return False
        return int(asp.rank[action]) > int(asp.rank[la])

    def unseen_counts(self) -> np.ndarray:
        # 未现牌 = 全牌 - 自己手牌 - 已出牌（含对手手里的底牌）
        unseen = TOTAL_COUNTS.astype(np.int16) - mask_to_counts(self.human_hand).astype(np.int16) - self.played.sum(0).astype(np.int16)
        return np.maximum(unseen, 0)

    def card_library(self) -> np.ndarray:
        return self.unseen_counts()

    # ---------- 推进 ----------
    def apply(self, player: int, action: int):
        asp = self.asp
        if self.winner >= 0:
            raise RuntimeError("game finished")
        if not self.is_legal_for(player, action):
            raise ValueError(f"illegal action {action} for player {player}")
        if action == asp.n:
            self.passes += 1
            self.history.append((player, -1))
            if self.passes >= 2:
                self.passes = 0
                self.last_act = -1
                self.turn = self.last_player if self.last_player >= 0 else (player + 1) % 3
            else:
                self.turn = (player + 1) % 3
            return
        mask = int(asp.mask[action])
        if player == self.human_seat:
            self.human_hand = canonicalize_mask(self.human_hand & ~mask)
        self.played[player] += asp.counts[action]
        self.hand_sizes[player] -= int(asp.ncards[action])
        self.last_act = action
        self.last_player = player
        self.passes = 0
        self.history.append((player, action))
        if self.hand_sizes[player] == 0:
            self.winner = player
        else:
            self.turn = (player + 1) % 3

    # ---------- 推理 ----------
    def _base_states(self, samples: int, rng):
        return known_states(self.human_hand, self.human_seat, self.played, self.bottom,
                            self.landlord, self.turn, self.last_act, self.last_player,
                            self.passes, self.history, self.hand_sizes, samples, rng)

    def policy_value(self, model, device, action_feat, samples: int = 32, rng=None) -> float:
        if self.winner >= 0:
            return 1.0 if win_flags(np.array([self.winner]), np.array([self.landlord]),
                                     self.human_seat)[0] else 0.0
        rng = rng or np.random.default_rng()
        st = self._base_states(samples, rng)
        eng = BatchEngine(samples, seed=int(rng.integers(1, 1 << 30)), auto_landlord=False)
        states_to_engine(eng, st)
        w = run_policy(eng, model, device, action_feat)
        flags = win_flags(w, st["landlord"], self.human_seat)
        return float(flags.mean())

    def camp_value(self, model, device, action_feat, samples: int = 12, rng=None):
        """当前局面的地主/农民胜率估计，返回 (p_landlord, p_farmer)。"""
        if self.winner >= 0:
            ll = 1.0 if self.winner == self.landlord else 0.0
            return ll, 1.0 - ll
        rng = rng or np.random.default_rng()
        st = self._base_states(samples, rng)
        eng = BatchEngine(samples, seed=int(rng.integers(1, 1 << 30)), auto_landlord=False)
        states_to_engine(eng, st)
        w = run_policy(eng, model, device, action_feat)
        ll = float((w == st["landlord"]).mean())
        return ll, 1.0 - ll

    def _q_values(self, model, device, action_feat, actions, rng):
        st = self._base_states(1, rng)
        eng = BatchEngine(1, seed=1, auto_landlord=False)
        states_to_engine(eng, st)
        s, h = eng.observe()
        idx, cnt = eng.legal()
        legal = [int(x) for x in idx[0, :cnt[0]]]
        s_t = torch.as_tensor(s, device=device)
        h_t = torch.as_tensor(h, device=device).view(1, HIST, HIST_DIM)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                    enabled=torch.device(device).type == "cuda"):
            z = model.encode(s_t, h_t)
            flat = torch.as_tensor(legal, dtype=torch.int64, device=device)
            q = model.q_from_z(z.expand(len(legal), -1), flat, action_feat[flat]).float().cpu().numpy()
        return legal, q

    def candidates(self, model, device, action_feat, topk: int = 3, samples: int = 64,
                   rng=None, time_budget_ms: float = 950, max_samples: int | None = None,
                   progress_cb=None, cancel_event=None, publish_ms: float = 200.0):
        """返回 (best_actions[(action, value)], policy_value)。

        策略：Q 网络先选出前 topk 个候选，再按 4→8→16→32→64→128 逐步增加
        rollout 样本；时间预算用完或达到 max_samples 即停（anytime）。
        """
        import time as _time
        t0 = _time.monotonic()
        self.last_search = {"elapsed_ms": 0.0, "samples": 0, "candidates": 0}
        if self.winner >= 0 or self.turn != self.human_seat:
            return [], 0.0
        rng = rng or np.random.default_rng()
        legal, q = self._q_values(model, device, action_feat, None, rng)
        order = np.argsort(-q)
        # 按 (类型, 主点数, 链长) 去重，保证前三选是不同“核心”动作，
        # 而不是同一个飞机的不同带牌。
        chosen = []
        seen = set()
        for oi in order:
            a = legal[int(oi)]
            if a == self.asp.n:
                core = ("pass",)
            else:
                core = (int(self.asp.typ[a]), int(self.asp.rank[a]), int(self.asp.chain[a]))
            if core in seen:
                continue
            seen.add(core)
            chosen.append(a)
            if len(chosen) >= topk:
                break
        if len(chosen) < min(topk, len(order)):
            for oi in order:
                a = legal[int(oi)]
                if a not in chosen:
                    chosen.append(a)
                if len(chosen) >= topk:
                    break
        pass_id = self.asp.n
        base_n = int(mask_to_counts(self.human_hand).sum())
        vals = [0.0] * len(chosen)
        stage_done = 0
        last_dur = 0.0
        last_pub = -1e9

        def publish():
            order2 = np.argsort(-np.array(vals))
            best = [(chosen[int(i)], float(vals[int(i)])) for i in order2]
            pol = float(max(vals)) if vals else 0.0
            if progress_cb:
                progress_cb(best[:3], pol, stage_done,
                            (_time.monotonic() - t0) * 1000)

        stage_list = [4, 8, 16, 32, 64, 128, 256, 512]
        limit = max_samples if max_samples is not None else samples
        for stage in stage_list:
            if cancel_event is not None and cancel_event.is_set():
                break
            if stage > limit:
                break
            elapsed = (_time.monotonic() - t0) * 1000
            if stage_done and elapsed + last_dur * 1.15 > time_budget_ms:
                break
            t1 = _time.monotonic()
            base = self._base_states(stage, rng)
            all_states = []
            immediate = []
            for a in chosen:
                if a != pass_id and int(self.asp.ncards[a]) == base_n:
                    immediate.append(True)
                    all_states.append(self._pad_state(base))
                else:
                    immediate.append(False)
                    all_states.append(self._apply_to_base(base, a))
            merged = {kk: np.concatenate([st[kk] for st in all_states], axis=0) for kk in base}
            eng = BatchEngine(len(chosen) * stage,
                              seed=int(rng.integers(1, 1 << 30)), auto_landlord=False)
            states_to_engine(eng, merged)
            w = run_policy(eng, model, device, action_feat)
            flags = win_flags(w, merged["landlord"], self.human_seat).reshape(len(chosen), stage)
            vals = [1.0 if immediate[i] else float(flags[i].mean()) for i in range(len(chosen))]
            last_dur = (_time.monotonic() - t1) * 1000
            stage_done = stage
            now_ms = (_time.monotonic() - t0) * 1000
            if progress_cb and (now_ms - last_pub >= publish_ms or now_ms >= time_budget_ms):
                publish()
                last_pub = now_ms
            if now_ms > time_budget_ms:
                break
        publish()
        order2 = np.argsort(-np.array(vals))
        best = [(chosen[int(i)], float(vals[int(i)])) for i in order2]
        self.last_search = {"elapsed_ms": (_time.monotonic() - t0) * 1000,
                            "samples": stage_done, "candidates": len(chosen)}
        pol = float(max(vals)) if vals else 0.0
        return best[:3], pol

    def _pad_state(self, base):
        return {k: v.copy() for k, v in base.items()}

    def _apply_to_base(self, base, action):
        asp = self.asp
        st = {k: np.array(v, copy=True) for k, v in base.items()}
        G = st["hands"].shape[0]
        human = self.human_seat
        pass_id = asp.n
        for i in range(G):
            if action == pass_id:
                st["passes"][i] = st["passes"][i] + 1
                if st["passes"][i] >= 2:
                    st["passes"][i] = 0
                    st["last_act"][i] = -1
                    lp = int(st["last_player"][i])
                    st["turn"][i] = lp if lp >= 0 else (human + 1) % 3
                else:
                    st["turn"][i] = (human + 1) % 3
                self._hist_append(st, i, -1, human)
            else:
                mask = int(asp.mask[action])
                st["hands"][i, human] = canonicalize_mask(int(st["hands"][i, human]) & ~mask)
                st["played"][i, human] += asp.counts[action]
                st["last_act"][i] = action
                st["last_player"][i] = human
                st["passes"][i] = 0
                st["turn"][i] = (human + 1) % 3
                self._hist_append(st, i, action, human)
        return st

    @staticmethod
    def _hist_append(st, i, action, player):
        hl = int(st["hist_len"][i])
        if hl < HIST:
            st["hist_a"][i, hl] = action
            st["hist_p"][i, hl] = player
            st["hist_len"][i] = hl + 1
        else:
            st["hist_a"][i, :-1] = st["hist_a"][i, 1:]
            st["hist_p"][i, :-1] = st["hist_p"][i, 1:]
            st["hist_a"][i, HIST - 1] = action
            st["hist_p"][i, HIST - 1] = player


def should_grab(p_ll: float, p_f: float) -> bool:
    """常规模式：胜率更高 / 地主胜率 >70% / 低于农民不超过 5%。"""
    return bool(p_ll > p_f or p_ll > 0.70 or p_ll >= p_f - 0.05)


NOSHUFFLE_GRAB_THRESHOLD = 0.75


def should_grab_noshuffle(p_ll: float) -> bool:
    """不洗牌模式更严格：地主胜率必须 >75% 才推荐抢。"""
    return bool(p_ll > NOSHUFFLE_GRAB_THRESHOLD)


def bid_advice(bid_net, hand_mask: int, device="cuda"):
    """返回 (p_landlord, p_farmer, 是否建议抢)。"""
    if bid_net is None:
        return 0.5, 0.5, True
    scores = score_hands(bid_net, np.array([int(hand_mask)], dtype=np.int64), device)
    p_ll, p_f = float(scores[0, 0]), float(scores[0, 1])
    return p_ll, p_f, should_grab(p_ll, p_f)


def bid_rollout(hand_mask: int, model, device, action_feat, samples: int = 32, seed: int = 0,
                asp=None):
    """用完整 rollout 精确估计 17 张手牌的地主/农民胜率。"""
    asp = asp or get_action_space()
    rng = np.random.default_rng(seed)
    out = {}
    for role in (0, 1):
        st = bid_states(np.array([int(hand_mask)], dtype=np.int64), role, samples, rng)
        eng = BatchEngine(samples, seed=int(rng.integers(1, 1 << 30)), auto_landlord=False)
        states_to_engine(eng, st)
        w = run_policy(eng, model, device, action_feat)
        flags = win_flags(w, st["landlord"], st["human_seat"])
        out[role] = float(flags.mean())
    return out[0], out[1]
