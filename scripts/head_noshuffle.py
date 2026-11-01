"""不洗牌牌局：常规队 vs 不洗牌队，整方头对头（同牌局，地主/农民互换）。"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch

from doudizhu.engine import BatchEngine, HIST
from doudizhu.model import load_checkpoint
from doudizhu.reference import get_action_space
from doudizhu.selfplay import SelfPlayRunner, select_actions_batch
from doudizhu.train import make_action_features
from doudizhu.bots import make_bot

ap = __import__("argparse").ArgumentParser()
ap.add_argument("--regular", default=str(ROOT / "checkpoints" / "best_model.pt"))
ap.add_argument("--noshuffle", default=str(ROOT / "checkpoints" / "best_model_noshuffle.pt"))
ap.add_argument("--deals", type=int, default=256)
ap.add_argument("--seed", type=int, default=2025)
ap.add_argument("--out", default=str(ROOT / "checkpoints" / "head_noshuffle.json"))
args = ap.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asp = get_action_space(); af_np = make_action_features(asp)
af = torch.as_tensor(af_np, dtype=torch.float32, device=device)
bot = make_bot("heuristic", seed=7)
rng = np.random.default_rng(args.seed)

# 生成不洗牌 deals（用启发式推进牌堆，和 eval_noshuffle 同一规则）
ge = BatchEngine(1, seed=args.seed, auto_landlord=False)
gr = SelfPlayRunner(ge, None, "cpu", af_np, no_shuffle=True, oracle_landlord=False)
deals = []
for _ in range(args.deals):
    hands = ge.hands.copy(); bottom = ge.bottom.copy(); ll = int(rng.integers(0, 3))
    deals.append((hands, bottom, ll))
    ge.assign_landlords(np.array([0], dtype=np.int32), np.array([ll], dtype=np.int8))
    steps = 0
    while ge.winner[0] < 0 and steps < 400:
        idx, cnt = ge.legal(); t = int(ge.turn[0])
        legal = idx[0, :int(cnt[0])].tolist()
        sizes = [int(ge.hands[0, p]).bit_count() for p in range(3)]
        a = bot.choose(int(ge.hands[0, t]), int(ge.last_act[0]), legal, t,
                       int(ge.landlord[0]), int(ge.last_player[0]), sizes)
        acts = np.array([a], dtype=np.int32)
        gr._record_discard(acts); ge.step(acts); steps += 1
    gr._finish_no_shuffle_deck(0)
    ge.reset_slots(np.array([0], dtype=np.int32)); gr._deal_no_shuffle(np.array([0], dtype=np.int32))

reg, _ = load_checkpoint(args.regular, map_location=device); reg.to(device).eval()
ns, _ = load_checkpoint(args.noshuffle, map_location=device); ns.to(device).eval()


def run_side(a_model, b_model, hands, bottom, ll, a_landlord):
    e = BatchEngine(1, seed=1, auto_landlord=False)
    e.set_states(hands, np.zeros((1, 3, 15), dtype=np.int8), bottom,
                 np.array([-1], dtype=np.int8), np.array([-1], dtype=np.int8),
                 np.array([-1], dtype=np.int32), np.array([-1], dtype=np.int8),
                 np.array([0], dtype=np.int8), np.full((1, HIST), -1, dtype=np.int32),
                 np.full((1, HIST), -1, dtype=np.int8), np.array([0], dtype=np.int8))
    e.assign_landlords(np.array([0], dtype=np.int32), np.array([ll], dtype=np.int8))
    steps = 0
    while e.winner[0] < 0 and steps < 400:
        st, hist = e.observe(); idx, cnt = e.legal()
        a_is = (int(e.turn[0]) == ll) == a_landlord
        m = a_model if a_is else b_model
        a = int(select_actions_batch(m, device, af, st, hist, idx, cnt, 0.0)[0])
        e.step(np.array([a], dtype=np.int32)); steps += 1
    w = int(e.winner[0])
    if w < 0:
        return None
    if a_landlord:
        return w == ll
    return w != ll

t0 = time.time(); wins = 0; games = 0; invalid = 0
for hands, bottom, ll in deals:
    # A=regular 当地主，B=专家两农民
    r = run_side(reg, ns, hands, bottom, ll, True)
    if r is None: invalid += 1
    else: wins += int(r); games += 1
    # A=regular 两农民，B=专家地主
    r = run_side(reg, ns, hands, bottom, ll, False)
    if r is None: invalid += 1
    else: wins += int(r); games += 1
out = {"deals": len(deals), "games": games, "invalid": invalid,
       "regular_side_win": wins / max(1, games), "elapsed_s": round(time.time()-t0,1)}
print(json.dumps(out, ensure_ascii=False), flush=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
