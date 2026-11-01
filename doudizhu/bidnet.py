"""叫地主/抢地主评估网络：输入 17 张手牌，输出地主/农民胜率。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .actions import N_RANKS, counts_to_mask, mask_to_counts

POP4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.int8)
TOTAL_COUNTS = np.array([4] * 13 + [1, 1], dtype=np.int8)


def hand_counts(masks: np.ndarray) -> np.ndarray:
    masks = np.asarray(masks, dtype=np.int64).reshape(-1)
    G = len(masks)
    cnt = np.zeros((G, N_RANKS), dtype=np.int8)
    for r in range(13):
        cnt[:, r] = POP4[((masks >> (4 * r)) & 15).astype(np.int64)]
    cnt[:, 13] = (masks >> 52) & 1
    cnt[:, 14] = (masks >> 53) & 1
    return cnt


def hand_features(masks: np.ndarray) -> np.ndarray:
    """17 张手牌的固定维特征（约 25 维）。"""
    cnt = hand_counts(masks).astype(np.float32)  # [G,15]
    G = cnt.shape[0]
    total = cnt.sum(1)
    bombs = (cnt[:, :13] >= 4).sum(1).astype(np.float32)
    rocket = ((cnt[:, 13] > 0) & (cnt[:, 14] > 0)).astype(np.float32)
    jokers = cnt[:, 13] + cnt[:, 14]
    twos = cnt[:, 12]
    aces = cnt[:, 11]
    high = twos + jokers + aces
    trios = (cnt[:, :13] >= 3).sum(1).astype(np.float32)
    pairs = (cnt[:, :13] == 2).sum(1).astype(np.float32)
    singles = (cnt[:, :13] == 1).sum(1).astype(np.float32)
    best = np.zeros(G, dtype=np.float32)
    run = np.zeros(G, dtype=np.float32)
    for r in range(12):
        has = (cnt[:, r] > 0).astype(np.float32)
        run = (run + 1.0) * has
        best = np.maximum(best, run)
    maxc = cnt[:, :13].max(1).astype(np.float32)
    feats = [
        cnt / 4.0,
        (total / 17.0)[:, None],
        (bombs / 2.0)[:, None],
        rocket[:, None],
        (jokers / 2.0)[:, None],
        (twos / 4.0)[:, None],
        (aces / 4.0)[:, None],
        (high / 8.0)[:, None],
        (trios / 4.0)[:, None],
        (pairs / 6.0)[:, None],
        (singles / 8.0)[:, None],
        (best / 12.0)[:, None],
        (maxc / 4.0)[:, None],
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)


FEATURE_DIM = hand_features(np.array([0], dtype=np.int64)).shape[1]


class BidNet(nn.Module):
    def __init__(self, in_dim: int = FEATURE_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


def build_bid_net() -> BidNet:
    return BidNet()


def save_bid_net(path: str, model: BidNet, meta: dict | None = None):
    torch.save({"model": model.state_dict(), "in_dim": model.net[0].in_features,
                "meta": meta or {}}, path)


def load_bid_net(path: str, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = BidNet(in_dim=payload["in_dim"])
    model.load_state_dict(payload["model"])
    model.eval()
    return model, payload


@torch.no_grad()
def score_hands(model: BidNet, masks: np.ndarray, device="cuda", batch: int = 4096):
    """返回 [N,2]：列 0 = 当地主胜率，列 1 = 当农民胜率。"""
    masks = np.asarray(masks, dtype=np.int64).reshape(-1)
    out = np.zeros((len(masks), 2), dtype=np.float32)
    device = torch.device(device)
    model = model.to(device).eval()
    for i in range(0, len(masks), batch):
        x = torch.as_tensor(hand_features(masks[i:i + batch]), device=device)
        out[i:i + batch] = model(x).float().cpu().numpy()
    return out
