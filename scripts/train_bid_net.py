"""训练叫地主评估网络。"""
import argparse, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn

from doudizhu.bidnet import FEATURE_DIM, build_bid_net, hand_features, save_bid_net

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=r"<repo>\runs\bid_data.npz")
ap.add_argument("--out", default=str(ROOT / "checkpoints" / "bid_net.pt"))
ap.add_argument("--epochs", type=int, default=300)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
args = ap.parse_args()

d = np.load(args.data, allow_pickle=True)
masks = d["masks"]
y = np.stack([d["p_landlord"], d["p_farmer"]], axis=1).astype(np.float32)
X = hand_features(masks)
print("data", X.shape, y.shape, "rollouts", int(d["rollouts"]))
perm = np.random.default_rng(0).permutation(len(X))
n_val = max(1, int(len(X) * 0.1))
val = perm[:n_val]
tr = perm[n_val:]
Xtr = torch.as_tensor(X[tr], device="cuda")
Ytr = torch.as_tensor(y[tr], device="cuda")
Xva = torch.as_tensor(X[val], device="cuda")
Yva = torch.as_tensor(y[val], device="cuda")
model = build_bid_net().to("cuda")
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
lossf = nn.MSELoss()
best = 1e9
best_state = None
for ep in range(args.epochs):
    model.train()
    idx = torch.randperm(len(Xtr), device="cuda")
    tot = 0.0
    for i in range(0, len(idx), args.batch):
        b = idx[i:i + args.batch]
        pred = model(Xtr[b])
        loss = lossf(pred, Ytr[b])
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss) * len(b)
    model.eval()
    with torch.no_grad():
        pv = model(Xva)
        vloss = float(lossf(pv, Yva))
        mae = float((pv - Yva).abs().mean())
        corr = float(torch.corrcoef(torch.stack([pv[:, 0], Yva[:, 0]]))[0, 1])
    if vloss < best:
        best = vloss
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if ep % 20 == 0 or ep == args.epochs - 1:
        print(f"ep {ep} train {tot/len(Xtr):.5f} val {vloss:.5f} mae {mae:.4f} corr_ll {corr:.4f}", flush=True)

model.load_state_dict(best_state)
os.makedirs(os.path.dirname(args.out), exist_ok=True)
save_bid_net(args.out, model, {"data": args.data, "n": len(X), "val_mse": best,
                               "rollouts": int(d["rollouts"]), "model": str(d["model"])})
print("saved", args.out, "best val", best)
