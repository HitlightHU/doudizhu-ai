import sys, argparse, json, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from doudizhu.bidnet import load_bid_net
from doudizhu.bots import make_bot
from doudizhu.evaluate import evaluate_vs_bot, evaluate_vs_bot_oracle, evaluate_team_vs_bot
from doudizhu.model import load_checkpoint
from doudizhu.reference import get_action_space
from doudizhu.train import make_action_features

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "checkpoints" / "best_model.pt"))
ap.add_argument("--bid-net", default=str(ROOT / "checkpoints" / "bid_net.pt"))
ap.add_argument("--games", type=int, default=256)
ap.add_argument("--seed", type=int, default=2025)
args = ap.parse_args()
asp = get_action_space(); device = torch.device("cuda")
feat = torch.as_tensor(make_action_features(asp), device=device)
model, _ = load_checkpoint(args.model, map_location=device); model.to(device).eval()
bid, _ = load_bid_net(args.bid_net, map_location=device); bid.to(device).eval()
t0 = time.time()
rand = evaluate_vs_bot(model, make_bot("random", seed=7), feat, device, n_games=args.games, seed=args.seed)
heur = evaluate_vs_bot(model, make_bot("heuristic", seed=7), feat, device, n_games=args.games, seed=args.seed)
orc = evaluate_vs_bot_oracle(model, bid, make_bot("heuristic", seed=7), feat, device,
                             n_games=args.games, seed=args.seed)
print("random", rand)
print("heuristic(random LL)", heur)
print("heuristic(oracle LL)", orc)
print("elapsed", round(time.time()-t0, 1))
