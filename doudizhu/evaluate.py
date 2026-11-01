"""评估：批量对局（模型 vs 随机 bot / 历史模型）。"""
from __future__ import annotations

import numpy as np
import torch

from .engine import BatchEngine
from .selfplay import select_actions_batch


def _win_flags(results: np.ndarray, landlord: np.ndarray, model_seat: int):
    is_ll = model_seat == landlord
    won = np.where(is_ll, results == model_seat, results != landlord)
    valid = results >= 0
    return won & valid, is_ll & valid


def evaluate_vs_random(model, action_feat, device, n_games: int = 512, seed: int = 1,
                       autocast_dtype=torch.bfloat16, max_steps: int = 400):
    """模型分别坐 3 个座位，其余两家随机出牌；固定发牌种子。"""
    model.eval()
    all_won, all_ll = [], []
    total = 0
    for seat in range(3):
        eng = BatchEngine(n_games, seed=seed)
        active = np.arange(n_games)
        res = -np.ones(n_games, dtype=np.int32)
        rng = np.random.default_rng(seed + 777 + seat)
        steps = 0
        while active.size and steps < max_steps:
            st, hist = eng.observe()
            idx, cnt = eng.legal()
            acts = np.zeros(n_games, dtype=np.int32)
            acnt = cnt[active]
            acts[active] = idx[active, (rng.random(active.size) * acnt).astype(np.int64)]
            m = active[eng.turn[active] == seat]
            if m.size:
                acts[m] = select_actions_batch(model, device, action_feat, st[m], hist[m],
                                               idx[m], cnt[m], 0.0, autocast_dtype)
            eng.step(acts)
            w = eng.winner
            newly = active[w[active] >= 0]
            res[newly] = w[newly]
            active = active[w[active] < 0]
            steps += 1
        won, is_ll = _win_flags(res, eng.landlord, seat)
        all_won.append(won)
        all_ll.append(is_ll)
        total += n_games
    won = np.concatenate(all_won)
    is_ll = np.concatenate(all_ll)
    out = {
        "games": int(won.size),
        "win": float(won.mean()),
        "landlord_games": int(is_ll.sum()),
        "landlord_win": float(won[is_ll].mean()) if is_ll.any() else float("nan"),
        "farmer_games": int((~is_ll).sum()),
        "farmer_win": float(won[~is_ll].mean()) if (~is_ll).any() else float("nan"),
    }
    return out


def evaluate_vs_bot_oracle(model, bid_net, bot, action_feat, device, n_games: int = 512,
                           seed: int = 1, autocast_dtype=torch.bfloat16, max_steps: int = 400):
    """地主由叫地主网络在三个人中选（上帝视角分配），其余同 evaluate_vs_bot。"""
    from .bidnet import score_hands
    model.eval()
    all_won, all_ll = [], []
    for seat in range(3):
        eng = BatchEngine(n_games, seed=seed, auto_landlord=False)
        slots = np.arange(n_games, dtype=np.int32)
        masks = eng.hands[slots].reshape(-1)
        scores = score_hands(bid_net, masks, device)
        eng.assign_landlords(slots, scores[:, 0].reshape(n_games, 3).argmax(1).astype(np.int8))
        active = np.arange(n_games)
        res = -np.ones(n_games, dtype=np.int32)
        steps = 0
        while active.size and steps < max_steps:
            st, hist = eng.observe()
            idx, cnt = eng.legal()
            acts = np.zeros(n_games, dtype=np.int32)
            m = active[eng.turn[active] == seat]
            o = active[eng.turn[active] != seat]
            if m.size:
                acts[m] = select_actions_batch(model, device, action_feat, st[m], hist[m],
                                               idx[m], cnt[m], 0.0, autocast_dtype)
            for g in o:
                t = int(eng.turn[g])
                legal = idx[g, :cnt[g]].tolist()
                sizes = [int(eng.hands[g, p]).bit_count() for p in range(3)]
                acts[g] = bot.choose(int(eng.hands[g, t]), int(eng.last_act[g]), legal, t,
                                     int(eng.landlord[g]), int(eng.last_player[g]), sizes)
            eng.step(acts)
            w = eng.winner
            newly = active[w[active] >= 0]
            res[newly] = w[newly]
            active = active[w[active] < 0]
            steps += 1
        won, is_ll = _win_flags(res, eng.landlord, seat)
        all_won.append(won)
        all_ll.append(is_ll)
    won = np.concatenate(all_won)
    is_ll = np.concatenate(all_ll)
    return {
        "games": int(won.size),
        "win": float(won.mean()),
        "landlord_games": int(is_ll.sum()),
        "landlord_win": float(won[is_ll].mean()) if is_ll.any() else float("nan"),
        "farmer_games": int((~is_ll).sum()),
        "farmer_win": float(won[~is_ll].mean()) if (~is_ll).any() else float("nan"),
    }


def evaluate_vs_bot(model, bot, action_feat, device, n_games: int = 512, seed: int = 1,
                    autocast_dtype=torch.bfloat16, max_steps: int = 400):
    """模型分别坐 3 个座位，其余两家为规则 bot。"""
    model.eval()
    all_won, all_ll = [], []
    for seat in range(3):
        eng = BatchEngine(n_games, seed=seed)
        active = np.arange(n_games)
        res = -np.ones(n_games, dtype=np.int32)
        steps = 0
        while active.size and steps < max_steps:
            st, hist = eng.observe()
            idx, cnt = eng.legal()
            acts = np.zeros(n_games, dtype=np.int32)
            m = active[eng.turn[active] == seat]
            o = active[eng.turn[active] != seat]
            if m.size:
                acts[m] = select_actions_batch(model, device, action_feat, st[m], hist[m],
                                               idx[m], cnt[m], 0.0, autocast_dtype)
            for g in o:
                t = int(eng.turn[g])
                legal = idx[g, :cnt[g]].tolist()
                sizes = [int(eng.hands[g, p]).bit_count() for p in range(3)]
                acts[g] = bot.choose(int(eng.hands[g, t]), int(eng.last_act[g]), legal, t,
                                     int(eng.landlord[g]), int(eng.last_player[g]), sizes)
            eng.step(acts)
            w = eng.winner
            newly = active[w[active] >= 0]
            res[newly] = w[newly]
            active = active[w[active] < 0]
            steps += 1
        won, is_ll = _win_flags(res, eng.landlord, seat)
        all_won.append(won)
        all_ll.append(is_ll)
    won = np.concatenate(all_won)
    is_ll = np.concatenate(all_ll)
    return {
        "games": int(won.size),
        "win": float(won.mean()),
        "landlord_games": int(is_ll.sum()),
        "landlord_win": float(won[is_ll].mean()) if is_ll.any() else float("nan"),
        "farmer_games": int((~is_ll).sum()),
        "farmer_win": float(won[~is_ll].mean()) if (~is_ll).any() else float("nan"),
    }


def evaluate_team_vs_bot(model, bot, action_feat, device, n_games: int = 512, seed: int = 1,
                         autocast_dtype=torch.bfloat16, max_steps: int = 400):
    """模型控制完整一方（地主，或两个农民），对手全部是规则 bot。"""
    model.eval()
    out = {}
    for side in ("landlord", "farmers"):
        eng = BatchEngine(n_games, seed=seed)
        active = np.arange(n_games)
        res = -np.ones(n_games, dtype=np.int32)
        steps = 0
        while active.size and steps < max_steps:
            st, hist = eng.observe()
            idx, cnt = eng.legal()
            acts = np.zeros(n_games, dtype=np.int32)
            turn = eng.turn
            if side == "landlord":
                is_model = turn == eng.landlord
            else:
                is_model = turn != eng.landlord
            m = active[is_model[active]]
            o = active[~is_model[active]]
            if m.size:
                acts[m] = select_actions_batch(model, device, action_feat, st[m], hist[m],
                                               idx[m], cnt[m], 0.0, autocast_dtype)
            for g in o:
                t = int(turn[g])
                legal = idx[g, :cnt[g]].tolist()
                sizes = [int(eng.hands[g, p]).bit_count() for p in range(3)]
                acts[g] = bot.choose(int(eng.hands[g, t]), int(eng.last_act[g]), legal, t,
                                     int(eng.landlord[g]), int(eng.last_player[g]), sizes)
            eng.step(acts)
            w = eng.winner
            newly = active[w[active] >= 0]
            res[newly] = w[newly]
            active = active[w[active] < 0]
            steps += 1
        valid = res >= 0
        if side == "landlord":
            won = (res == eng.landlord) & valid
        else:
            won = (res != eng.landlord) & valid
        out[side + "_games"] = int(valid.sum())
        out[side + "_win"] = float(won[valid].mean()) if valid.any() else float("nan")
    out["win"] = float(np.nanmean([out["landlord_win"], out["farmers_win"]]))
    return out


def evaluate_vs_model(model_a, model_b, action_feat, device, n_games: int = 512,
                      seed: int = 1, autocast_dtype=torch.bfloat16, max_steps: int = 400):
    """模型 A 坐 3 个座位分别对战模型 B（其余两家）。"""
    model_a.eval()
    model_b.eval()
    all_won, all_ll = [], []
    for seat in range(3):
        eng = BatchEngine(n_games, seed=seed)
        active = np.arange(n_games)
        res = -np.ones(n_games, dtype=np.int32)
        steps = 0
        while active.size and steps < max_steps:
            st, hist = eng.observe()
            idx, cnt = eng.legal()
            acts = np.zeros(n_games, dtype=np.int32)
            m = active[eng.turn[active] == seat]
            o = active[eng.turn[active] != seat]
            if m.size:
                acts[m] = select_actions_batch(model_a, device, action_feat, st[m], hist[m],
                                               idx[m], cnt[m], 0.0, autocast_dtype)
            if o.size:
                acts[o] = select_actions_batch(model_b, device, action_feat, st[o], hist[o],
                                               idx[o], cnt[o], 0.0, autocast_dtype)
            eng.step(acts)
            w = eng.winner
            newly = active[w[active] >= 0]
            res[newly] = w[newly]
            active = active[w[active] < 0]
            steps += 1
        won, is_ll = _win_flags(res, eng.landlord, seat)
        all_won.append(won)
        all_ll.append(is_ll)
    won = np.concatenate(all_won)
    is_ll = np.concatenate(all_ll)
    return {
        "games": int(won.size),
        "win": float(won.mean()),
        "landlord_games": int(is_ll.sum()),
        "landlord_win": float(won[is_ll].mean()) if is_ll.any() else float("nan"),
        "farmer_games": int((~is_ll).sum()),
        "farmer_win": float(won[~is_ll].mean()) if (~is_ll).any() else float("nan"),
    }
