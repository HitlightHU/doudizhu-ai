"""最终评估：随机 / 贪心 / 启发式 / 团队模式 / 对旧模型。"""
import argparse, json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from doudizhu.bots import make_bot
from doudizhu.evaluate import (evaluate_team_vs_bot, evaluate_vs_bot, evaluate_vs_model,
                               evaluate_vs_random)
from doudizhu.model import load_checkpoint
from doudizhu.reference import get_action_space
from doudizhu.train import make_action_features

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--opponent-model", default="")
ap.add_argument("--games", type=int, default=512)
ap.add_argument("--seed", type=int, default=2025)
ap.add_argument("--out", default="")
args = ap.parse_args()
asp = get_action_space()
device = torch.device("cuda")
feat = torch.as_tensor(make_action_features(asp), device=device)
model, payload = load_checkpoint(args.model, map_location=device)
model.to(device).eval()
meta = payload.get("meta", {})
res = {"model": args.model, "meta": {k: v for k, v in meta.items() if k != "config"},
       "games_per_mode": args.games, "seed": args.seed}
t0 = time.time()
res["vs_random"] = evaluate_vs_random(model, feat, device, n_games=args.games, seed=args.seed)
res["vs_greedy"] = evaluate_vs_bot(model, make_bot("greedy", seed=7), feat, device,
                                   n_games=args.games, seed=args.seed)
res["vs_heuristic"] = evaluate_vs_bot(model, make_bot("heuristic", seed=7), feat, device,
                                      n_games=args.games, seed=args.seed)
res["team_vs_heuristic"] = evaluate_team_vs_bot(model, make_bot("heuristic", seed=7), feat,
                                                device, n_games=args.games, seed=args.seed)
if args.opponent_model:
    opp, _ = load_checkpoint(args.opponent_model, map_location=device)
    opp.to(device).eval()
    res["vs_old_model"] = evaluate_vs_model(model, opp, feat, device, n_games=args.games,
                                            seed=args.seed)
res["elapsed_s"] = round(time.time() - t0, 1)

def row(name, d):
    return (f"| {name} | {d.get('games', d.get('landlord_games',0)+d.get('farmers_games',0)+d.get('farmer_games',0))} | "
            f"{d['win']:.3f} | {d.get('landlord_win', float('nan')):.3f} | "
            f"{d.get('farmer_win', d.get('farmers_win', float('nan'))):.3f} |")

print(f"# {args.model}")
print(f"meta: {res['meta']}")
print("| 模式 | 局数 | 总胜率 | 地主胜率 | 农民胜率 |")
print("|---|---|---|---|---|")
print(row("vs 随机", res["vs_random"]))
print(row("vs 贪心", res["vs_greedy"]))
print(row("vs 启发式(单座)", res["vs_heuristic"]))
print(row("vs 启发式(整方)", res["team_vs_heuristic"]))
if "vs_old_model" in res:
    print(row("vs 旧模型(单座)", res["vs_old_model"]))
print("elapsed", res["elapsed_s"], "s")
if args.out:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("saved", args.out)
