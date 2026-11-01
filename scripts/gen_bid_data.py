"""用当前模型 rollout 生成叫地主数据：每个 17 张手牌当地主/当农民的胜率。"""
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from doudizhu.actions import counts_to_mask, mask_to_counts
from doudizhu.bidnet import TOTAL_COUNTS, hand_counts
from doudizhu.engine import BatchEngine
from doudizhu.model import load_checkpoint
from doudizhu.reference import get_action_space
from doudizhu.rollout import bid_states, run_policy, states_to_engine
from doudizhu.train import make_action_features

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "checkpoints" / "best_model.pt"))
ap.add_argument("--hands", type=int, default=20000)
ap.add_argument("--rollouts", type=int, default=16)
ap.add_argument("--batch-hands", type=int, default=32)
ap.add_argument("--out", default=r"<repo>\runs\bid_data.npz")
ap.add_argument("--seed", type=int, default=2026)
args = ap.parse_args()

asp = get_action_space()
device = torch.device("cuda")
feat = torch.as_tensor(make_action_features(asp), device=device)
model, _ = load_checkpoint(args.model, map_location=device)
model.to(device).eval()
print("model loaded", args.model, flush=True)

rng = np.random.default_rng(args.seed)
deck_base = np.repeat(np.arange(15, dtype=np.int64), TOTAL_COUNTS.astype(np.int64))
seen = {}
masks = []
while len(masks) < args.hands:
    d = deck_base.copy()
    rng.shuffle(d)
    hand = d[:17]
    m = counts_to_mask(np.bincount(hand, minlength=15).astype(np.int8))
    if m in seen:
        continue
    seen[m] = 1
    masks.append(m)
masks = np.array(masks, dtype=np.int64)
counts = hand_counts(masks).astype(np.int8)
print("unique hands", len(masks), flush=True)

p_ll = np.zeros(args.hands, dtype=np.float32)
p_farmer = np.zeros(args.hands, dtype=np.float32)
t0 = time.time()
done = 0
for start in range(0, args.hands, args.batch_hands):
    end = min(args.hands, start + args.batch_hands)
    n = end - start
    for role, out in ((0, p_ll), (1, p_farmer)):
        st = bid_states(masks[start:end], role, args.rollouts, rng)
        G = st["hands"].shape[0]
        eng = BatchEngine(G, seed=int(rng.integers(1, 1 << 30)), auto_landlord=False)
        states_to_engine(eng, st)
        winners = run_policy(eng, model, device, feat)
        human_seat = st["human_seat"]
        wins = np.where(human_seat == st["landlord"], winners == human_seat,
                        winners != st["landlord"])
        out[start:end] = wins.reshape(n, args.rollouts).mean(1)
    done += n
    if (start // args.batch_hands) % 10 == 0:
        el = time.time() - t0
        games = done * args.rollouts * 2
        print(f"{done}/{args.hands} hands, {games} games, {games/max(el,1e-9):.0f} games/s, "
              f"elapsed {el:.0f}s", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
np.savez_compressed(args.out, masks=masks, counts=counts, p_landlord=p_ll,
                    p_farmer=p_farmer, rollouts=args.rollouts, model=args.model)
print("saved", args.out, "mean p_ll", p_ll.mean(), "mean p_farmer", p_farmer.mean())
