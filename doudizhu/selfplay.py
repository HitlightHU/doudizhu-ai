"""批量自对弈数据生成：向量化环境 + GPU 批量动作打分 + epsilon 探索。"""
from __future__ import annotations

import numpy as np
import torch

from .actions import N_RANKS, counts_to_mask, mask_to_counts
from .bidnet import score_hands
from .engine import HIST, HIST_DIM, MAXA, STATE_DIM

RING = 320


@torch.inference_mode()
def select_actions_batch(model, device, action_feat, st, hist, idx, cnt,
                         epsilon: float = 0.0, autocast_dtype=torch.bfloat16):
    """对一批决策点计算 argmax / epsilon-greedy 动作。"""
    device = torch.device(device)
    G = int(cnt.shape[0])
    st_t = torch.as_tensor(st, device=device)
    hist_t = torch.as_tensor(hist, device=device).view(G, HIST, HIST_DIM)
    valid = np.arange(MAXA, dtype=np.int32)[None, :] < cnt[:, None]
    flat = idx[valid].astype(np.int64)
    flat_t = torch.as_tensor(flat, device=device)
    cnt_t = torch.as_tensor(cnt, dtype=torch.int64, device=device)
    offsets = np.cumsum(cnt, dtype=np.int64) - cnt
    gid = np.repeat(np.arange(G, dtype=np.int64), cnt)
    gid_t = torch.as_tensor(gid, device=device)
    use_cuda = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_cuda):
        z = model.encode(st_t, hist_t)
        q = model.q_from_z(z[gid_t], flat_t, action_feat[flat_t]).float()
        qmax = torch.full((G,), -1e9, dtype=torch.float32, device=device)
        qmax.scatter_reduce_(0, gid_t, q, reduce="amax", include_self=True)
        best = q == qmax[gid_t]
        pos = torch.arange(flat.shape[0], dtype=torch.int64, device=device)
        best_pos = torch.full((G,), flat.shape[0], dtype=torch.int64, device=device)
        best_pos.scatter_reduce_(0, gid_t[best], pos[best], reduce="amin", include_self=True)
        best_act = flat_t[best_pos]
        if epsilon > 0:
            off_t = torch.as_tensor(offsets, device=device)
            rand_pos = off_t + (torch.rand(G, device=device) * cnt_t).long()
            rand_act = flat_t[rand_pos]
            use_rand = torch.rand(G, device=device) < epsilon
            acts = torch.where(use_rand, rand_act, best_act)
        else:
            acts = best_act
    return acts.to(torch.int32).cpu().numpy()


class SelfPlayRunner:
    def __init__(self, engine, model, device, action_feat: np.ndarray, ring: int = RING,
                 autocast_dtype=torch.bfloat16, bot=None, bot_prob: float = 0.0, seed: int = 123,
                 bid_net=None, oracle_landlord: bool = False, oracle_prob: float = 1.0,
                 pool=None, pool_prob: float = 0.0, no_shuffle: bool = False):
        self.engine = engine
        self.model = model
        self.device = torch.device(device)
        self.G = engine.G
        self.ring = int(ring)
        self.action_feat = torch.as_tensor(action_feat, dtype=torch.float32, device=self.device)
        self.trace_state = np.zeros((self.G, self.ring, STATE_DIM), dtype=np.float16)
        self.trace_hist = np.zeros((self.G, self.ring, HIST * HIST_DIM), dtype=np.float16)
        self.trace_act = np.zeros((self.G, self.ring), dtype=np.int32)
        self.trace_player = np.zeros((self.G, self.ring), dtype=np.int8)
        self.trace_bot = np.zeros((self.G, self.ring), dtype=bool)
        self.ep_start = np.zeros(self.G, dtype=np.int64)
        self.ep_len = np.zeros(self.G, dtype=np.int32)
        self.t = 0
        self.autocast_dtype = autocast_dtype
        self.generated = 0
        self.bot = bot
        self.bot_prob = float(bot_prob)
        self.bot_rng = np.random.default_rng(seed)
        self.bot_seat = np.full(self.G, -1, dtype=np.int8)
        self.pool = pool
        self.pool_prob = float(pool_prob)
        self.pool_idx = np.full(self.G, -1, dtype=np.int8)
        self.pool_seat = np.full(self.G, -1, dtype=np.int8)
        self.pool_rng = np.random.default_rng(seed + 31)
        self.bid_net = bid_net
        self.oracle_landlord = bool(oracle_landlord and bid_net is not None)
        self.oracle_prob = float(oracle_prob)
        self.bid_rng = np.random.default_rng(seed + 77)
        self.no_shuffle = bool(no_shuffle)
        self.deal_rng = np.random.default_rng(seed + 101)
        self.decks = np.zeros((self.G, 54), dtype=np.int8)
        self.discard = np.zeros((self.G, 54), dtype=np.int8)
        self.discard_len = np.zeros(self.G, dtype=np.int16)
        if self.no_shuffle:
            self._init_no_shuffle_decks()
            self._deal_no_shuffle(np.arange(self.G, dtype=np.int32))
        if self.oracle_landlord and getattr(engine, "auto_landlord", False):
            raise ValueError("oracle_landlord requires BatchEngine(auto_landlord=False)")
        if self.oracle_landlord:
            self._oracle_assign(np.arange(self.G, dtype=np.int32))
        self._assign_bots(np.arange(self.G, dtype=np.int32))
        self._assign_pool(np.arange(self.G, dtype=np.int32))

    def _init_no_shuffle_decks(self):
        base = np.array([r for r in range(13) for _ in range(4)] + [13, 14], dtype=np.int8)
        for g in range(self.G):
            self.decks[g] = self.deal_rng.permutation(base)
        self.discard_len[:] = 0
        self.discard[:, :] = 0

    def _deal_no_shuffle(self, slots: np.ndarray):
        """切牌后连续取三份 17 张，随机分给三个座位；最后 3 张做底牌。"""
        e = self.engine
        for g in slots:
            cut = int(self.deal_rng.integers(0, 54))
            d = np.concatenate([self.decks[g, cut:], self.decks[g, :cut]])
            blocks = [d[0:17], d[17:34], d[34:51]]
            order = self.deal_rng.permutation(3)
            for p in range(3):
                block = blocks[int(order[p])]
                cnt = np.bincount(block, minlength=N_RANKS).astype(np.int8)
                e.hands[g, p] = counts_to_mask(cnt)
            bcnt = np.bincount(d[51:54], minlength=N_RANKS).astype(np.int8)
            e.bottom[g] = counts_to_mask(bcnt)
            e.landlord[g] = -1
            e.turn[g] = -1
            e.last_act[g] = -1
            e.last_player[g] = -1
            e.passes[g] = 0
            e.winner[g] = -1
            e.hist_len[g] = 0
            self.discard_len[g] = 0

    def _record_discard(self, acts: np.ndarray):
        """按动作顺序记录本局打出的牌，供下一局不洗牌回收。"""
        if not self.no_shuffle:
            return
        e = self.engine
        for g in range(self.G):
            if e.winner[g] >= 0:
                continue
            a = int(acts[g])
            if a == int(e.asp.n):
                continue
            cnt = e.act_counts[a]
            n = int(self.discard_len[g])
            k = int(cnt.sum())
            if n + k > 54:
                continue
            self.discard[g, n:n + k] = np.repeat(
                np.arange(N_RANKS, dtype=np.int8), cnt.astype(np.int64))
            self.discard_len[g] = n + k

    def _finish_no_shuffle_deck(self, g: int):
        """出牌顺序原样保留；局末再接三家剩余手牌（各自按点数排序）。"""
        d = np.zeros(54, dtype=np.int8)
        n = int(self.discard_len[g])
        d[:n] = self.discard[g, :n]
        for p in range(3):
            cnt = mask_to_counts(int(self.engine.hands[g, p])).astype(np.int64)
            k = int(cnt.sum())
            if n + k > 54:
                break
            d[n:n + k] = np.repeat(np.arange(N_RANKS, dtype=np.int8), cnt)
            n += k
        if n == 54:
            self.decks[g] = d

    def _assign_pool(self, slots: np.ndarray):
        self.pool_idx[slots] = -1
        self.pool_seat[slots] = -1
        if self.pool is None or self.pool_prob <= 0 or not self.pool.models:
            return
        r = self.pool_rng.random(len(slots))
        sel = slots[(r < self.pool_prob) & (self.bot_seat[slots] < 0)]
        if sel.size:
            self.pool_idx[sel] = self.pool_rng.integers(0, len(self.pool.models), size=sel.size).astype(np.int8)
            self.pool_seat[sel] = self.pool_rng.integers(0, 3, size=sel.size).astype(np.int8)

    def _oracle_assign(self, slots: np.ndarray):
        if not self.oracle_landlord:
            return
        slots = np.asarray(slots, dtype=np.int32)
        if self.oracle_prob < 1.0 and len(slots):
            m = self.bid_rng.random(len(slots)) < self.oracle_prob
            oracle_slots = slots[m]
            rand_slots = slots[~m]
            if rand_slots.size:
                self.engine.assign_landlords(
                    rand_slots, self.bid_rng.integers(0, 3, size=len(rand_slots)).astype(np.int8))
        else:
            oracle_slots = slots
        if oracle_slots.size == 0:
            return
        masks = self.engine.hands[oracle_slots].reshape(-1)
        scores = score_hands(self.bid_net, masks, self.device)
        p_ll = scores[:, 0].reshape(len(oracle_slots), 3)
        choice = p_ll.argmax(axis=1).astype(np.int8)
        self.engine.assign_landlords(oracle_slots, choice)

    def _assign_bots(self, slots: np.ndarray):
        self.bot_seat[slots] = -1
        if self.bot is None or self.bot_prob <= 0:
            return
        r = self.bot_rng.random(len(slots))
        sel = slots[r < self.bot_prob]
        if sel.size:
            self.bot_seat[sel] = self.bot_rng.integers(0, 3, size=sel.size).astype(np.int8)

    def _autocast(self):
        use = self.device.type == "cuda"
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype, enabled=use)

    def select_actions(self, st, hist, idx, cnt, epsilon: float):
        return select_actions_batch(self.model, self.device, self.action_feat,
                                    st, hist, idx, cnt, epsilon, self.autocast_dtype)

    def _flush(self, done: np.ndarray, buffer, t: int):
        slots = np.nonzero(done)[0]
        for g in slots:
            L = int(self.ep_len[g])
            if L <= 0:
                continue
            start = int(self.ep_start[g])
            if L > self.ring:
                start += L - self.ring
                L = self.ring
            rows = (start + np.arange(L, dtype=np.int64)) % self.ring
            keep = ~self.trace_bot[g, rows]
            if not keep.any():
                continue
            rows = rows[keep]
            states = self.trace_state[g, rows]
            hists = self.trace_hist[g, rows]
            acts = self.trace_act[g, rows]
            players = self.trace_player[g, rows]
            win = int(self.engine.winner[g])
            landlord = int(self.engine.landlord[g])
            rr = 1.0 if win == landlord else -1.0
            rew = np.where(players == landlord, rr, -rr).astype(np.float16)
            buffer.add(states, hists, acts, rew)
            self.generated += 1
        slots_i = slots.astype(np.int32)
        if self.no_shuffle:
            for g in slots_i:
                self._finish_no_shuffle_deck(int(g))
        self.engine.reset_slots(slots_i)
        if self.no_shuffle:
            self._deal_no_shuffle(slots_i)
        if self.oracle_landlord:
            self._oracle_assign(slots_i)
        self._assign_bots(slots.astype(np.int32))
        self._assign_pool(slots.astype(np.int32))
        self.ep_start[slots] = t + 1
        self.ep_len[slots] = 0

    def run(self, n_games: int, epsilon: float, buffer):
        completed = 0
        G = self.G
        t = self.t
        while completed < n_games:
            st, hist = self.engine.observe()
            idx, cnt = self.engine.legal()
            acts = self.select_actions(st, hist, idx, cnt, epsilon)
            turn = self.engine.turn
            if self.bot is not None:
                bg = np.nonzero((self.bot_seat >= 0) & (turn == self.bot_seat))[0]
                for g in bg:
                    p = int(turn[g])
                    legal = idx[g, :cnt[g]].tolist()
                    sizes = [int(self.engine.hands[g, x]).bit_count() for x in range(3)]
                    acts[g] = self.bot.choose(int(self.engine.hands[g, p]),
                                              int(self.engine.last_act[g]), legal, p,
                                              int(self.engine.landlord[g]),
                                              int(self.engine.last_player[g]), sizes)
            if self.pool is not None:
                for k, pm in enumerate(self.pool.models):
                    pg = np.nonzero((self.pool_idx == k) & (turn == self.pool_seat))[0]
                    if pg.size:
                        acts[pg] = select_actions_batch(pm, self.device, self.action_feat,
                                                        st[pg], hist[pg], idx[pg], cnt[pg], 0.0)
            slot = t % self.ring
            self.trace_state[:, slot, :] = st
            self.trace_hist[:, slot, :] = hist
            self.trace_act[:, slot] = acts
            self.trace_player[:, slot] = turn
            self.trace_bot[:, slot] = ((self.bot_seat >= 0) & (turn == self.bot_seat)) | \
                                     ((self.pool_idx >= 0) & (turn == self.pool_seat))
            self.ep_len += 1
            self._record_discard(acts)
            self.engine.step(acts)
            win = self.engine.winner
            done = win >= 0
            nd = int(done.sum())
            if nd:
                self._flush(done, buffer, t)
                completed += nd
            t += 1
        self.t = t
        return completed
