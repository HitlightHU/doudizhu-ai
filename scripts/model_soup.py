"""检查点权重平均（model soup）。"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from doudizhu.model import load_checkpoint, save_checkpoint
from doudizhu.reference import get_action_space
from doudizhu.train import make_action_features

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
device = "cpu"
acc = None
model = None
for p in args.ckpts:
    m, payload = load_checkpoint(p, map_location=device)
    sd = m.state_dict()
    if acc is None:
        model = m
        acc = {k: v.float().clone() for k, v in sd.items()}
    else:
        for k, v in sd.items():
            acc[k] += v.float()
    print("loaded", p, payload.get("meta", {}).get("iter"))
n = len(args.ckpts)
for k in acc:
    acc[k] /= n
model.load_state_dict({k: v for k, v in acc.items()})
save_checkpoint(args.out, model, None, {"soup": args.ckpts, "n": n})
print("saved", args.out)
