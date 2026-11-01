"""测量 anytime 前三选的首个结果、128 样本和 512 样本时间。"""
import argparse, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, torch

from doudizhu.analysis import AnalysisSession
from doudizhu.model import load_checkpoint
from doudizhu.reference import Game, get_action_space
from doudizhu.train import make_action_features

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "checkpoints" / "best_model.pt"))
ap.add_argument("--n", type=int, default=16)
ap.add_argument("--out", default="")
args = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, _ = load_checkpoint(args.model, map_location=device)
model.to(device).eval()
asp = get_action_space()
af = torch.as_tensor(make_action_features(asp), dtype=torch.float32, device=device)

# GUI 启动时会预热；这里同样先跑一个小搜索，避免把首次 CUDA/Numba 初始化算进延迟
wg = Game.deal(seed=999)
ws = AnalysisSession.start(wg.hands[0], True, wg.bottom, 0)
ws.candidates(model, device, af, topk=1, time_budget_ms=1000, max_samples=8)

firsts, t64, t128, finals, samples = [], [], [], [], []
for i in range(args.n):
    g = Game.deal(seed=1000 + i)
    s = AnalysisSession.start(g.hands[0], True, g.bottom, 0)
    row = {"first": None, "64": None, "128": None, "final": None, "samples": 0}

    def cb(best, pol, n, elapsed):
        if row["first"] is None:
            row["first"] = elapsed
        if n >= 64 and row["64"] is None:
            row["64"] = elapsed
        if n >= 128 and row["128"] is None:
            row["128"] = elapsed
        row["final"] = elapsed
        row["samples"] = n

    s.candidates(model, device, af, topk=3, time_budget_ms=6000, max_samples=512,
                 progress_cb=cb, publish_ms=0)
    firsts.append(row["first"] or 0)
    t64.append(row["64"] or row["final"] or 0)
    t128.append(row["128"] or row["final"] or 0)
    finals.append(row["final"] or 0)
    samples.append(row["samples"])

def pct(a, q):
    return float(np.percentile(a, q))

print(f"n={args.n} first_ms p50={pct(firsts,50):.0f} p99={pct(firsts,99):.0f} max={max(firsts):.0f}")
print(f"at64_ms p50={pct(t64,50):.0f} p99={pct(t64,99):.0f} max={max(t64):.0f}")
print(f"at128_ms p50={pct(t128,50):.0f} p99={pct(t128,99):.0f} max={max(t128):.0f}")
print(f"final_ms p50={pct(finals,50):.0f} p99={pct(finals,99):.0f} max={max(finals):.0f}")
print(f"final_samples={sorted(set(samples))} under1200_at64={sum(x < 1200 for x in t64)}/{args.n} "
      f"under1200_at128={sum(x < 1200 for x in t128)}/{args.n}")
if args.out:
    import json
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"n": args.n, "first_ms": firsts, "at64_ms": t64, "at128_ms": t128,
                   "final_ms": finals, "samples": samples}, f, ensure_ascii=False, indent=2)
