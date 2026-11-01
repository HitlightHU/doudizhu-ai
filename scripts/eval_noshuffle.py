"""在不洗牌发牌分布上评估常规/不洗牌模型（随机出牌生成牌堆，模型对启发式 bot）。"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch

from doudizhu.bots import make_bot
from doudizhu.engine import BatchEngine, HIST
from doudizhu.model import load_checkpoint
from doudizhu.reference import get_action_space
from doudizhu.selfplay import SelfPlayRunner, select_actions_batch
from doudizhu.train import make_action_features

ap = argparse.ArgumentParser()
ap.add_argument("--regular", default=str(ROOT / "checkpoints" / "best_model.pt"))
ap.add_argument("--noshuffle", default=str(ROOT / "checkpoints" / "best_model_noshuffle.pt"))
ap.add_argument("--deals", type=int, default=128)
ap.add_argument("--seed", type=int, default=2025)
ap.add_argument("--out", default="")
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asp = get_action_space()
af_np = make_action_features(asp)
af = torch.as_tensor(af_np, dtype=torch.float32, device=device)
bot = make_bot("heuristic", seed=7)
rng = np.random.default_rng(args.seed)

# 用启发式出牌推进不洗牌牌堆（会打对子/顺子，更接近真实回收），保存每局初始手牌/底牌/随机地主
gen_engine = BatchEngine(1, seed=args.seed, auto_landlord=False)
gen_runner = SelfPlayRunner(gen_engine, None, "cpu", af_np, no_shuffle=True,
                            oracle_landlord=False)
deals = []
bomb_hands = 0
start = time.time()
for i in range(args.deals):
    hands = gen_engine.hands.copy()
    bottom = gen_engine.bottom.copy()
    ll = int(rng.integers(0, 3))
    deals.append((hands, bottom, ll))
    for p in range(3):
        h = int(hands[0, p])
        bomb_hands += sum(1 for r in range(13) if ((h >> (4 * r)) & 15) == 15)
    gen_engine.assign_landlords(np.array([0], dtype=np.int32),
                                np.array([ll], dtype=np.int8))
    steps = 0
    while gen_engine.winner[0] < 0 and steps < 400:
        idx, cnt = gen_engine.legal()
        t = int(gen_engine.turn[0])
        legal = idx[0, :int(cnt[0])].tolist()
        sizes = [int(gen_engine.hands[0, p]).bit_count() for p in range(3)]
        a = bot.choose(int(gen_engine.hands[0, t]), int(gen_engine.last_act[0]), legal, t,
                       int(gen_engine.landlord[0]), int(gen_engine.last_player[0]), sizes)
        acts = np.array([a], dtype=np.int32)
        gen_runner._record_discard(acts)
        gen_engine.step(acts)
        steps += 1
    gen_runner._finish_no_shuffle_deck(0)
    gen_engine.reset_slots(np.array([0], dtype=np.int32))
    gen_runner._deal_no_shuffle(np.array([0], dtype=np.int32))
print(f"generated {len(deals)} no-shuffle deals in {time.time()-start:.1f}s; "
      f"bombs/hand={bomb_hands / max(1, len(deals) * 3):.2f}", flush=True)


def run_model(model, hands, bottom, ll, seat):
    e = BatchEngine(1, seed=1, auto_landlord=False)
    e.set_states(hands, np.zeros((1, 3, 15), dtype=np.int8), bottom, np.array([-1], dtype=np.int8),
                 np.array([-1], dtype=np.int8), np.array([-1], dtype=np.int32),
                 np.array([-1], dtype=np.int8), np.array([0], dtype=np.int8),
                 np.full((1, HIST), -1, dtype=np.int32), np.full((1, HIST), -1, dtype=np.int8),
                 np.array([0], dtype=np.int8))
    e.assign_landlords(np.array([0], dtype=np.int32), np.array([ll], dtype=np.int8))
    steps = 0
    while e.winner[0] < 0 and steps < 400:
        st, hist = e.observe()
        idx, cnt = e.legal()
        legal = idx[0, :int(cnt[0])].tolist()
        if int(e.turn[0]) == seat:
            a = int(select_actions_batch(model, device, af, st, hist,
                                         idx, cnt, 0.0)[0])
        else:
            t = int(e.turn[0])
            sizes = [int(e.hands[0, p]).bit_count() for p in range(3)]
            a = bot.choose(int(e.hands[0, t]), int(e.last_act[0]), legal, t,
                           int(e.landlord[0]), int(e.last_player[0]), sizes)
        e.step(np.array([a], dtype=np.int32))
        steps += 1
    return int(e.winner[0])


def evaluate(model, tag):
    won, ll_games, ll_wins, f_games, f_wins, invalid = 0, 0, 0, 0, 0, 0
    t0 = time.time()
    for hands, bottom, ll in deals:
        for seat in range(3):
            w = run_model(model, hands, bottom, ll, seat)
            if w < 0:
                invalid += 1
                continue
            is_ll = seat == ll
            win = (w == seat) if is_ll else (w != ll)
            won += int(win)
            if is_ll:
                ll_games += 1; ll_wins += int(win)
            else:
                f_games += 1; f_wins += int(win)
    out = {"games": ll_games + f_games, "invalid": invalid,
           "win": won / max(1, ll_games + f_games),
           "landlord_games": ll_games, "landlord_win": ll_wins / max(1, ll_games),
           "farmer_games": f_games, "farmer_win": f_wins / max(1, f_games),
           "elapsed_s": round(time.time() - t0, 1)}
    print(tag, json.dumps(out, ensure_ascii=False), flush=True)
    return out

reg, _ = load_checkpoint(args.regular, map_location=device)
reg.to(device).eval()
res = {"regular": evaluate(reg, "regular")}
if args.noshuffle and __import__("os").path.exists(args.noshuffle):
    ns, _ = load_checkpoint(args.noshuffle, map_location=device)
    ns.to(device).eval()
    res["noshuffle"] = evaluate(ns, "noshuffle")
if args.out:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved", args.out)
