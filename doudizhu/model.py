"""动作打分式 Q 网络：状态编码 + 历史 LSTM + 动作编码 + Q 头。"""
from __future__ import annotations

import torch
import torch.nn as nn

from .actions import ACTION_FEAT_DIM
from .engine import HIST, HIST_DIM, STATE_DIM


DEFAULT_CFG = dict(
    state_dim=STATE_DIM,
    hist_dim=HIST_DIM,
    hist_len=HIST,
    action_feat_dim=ACTION_FEAT_DIM,
    state_hidden=256,
    lstm_hidden=128,
    action_hidden=64,
    q_hidden=128,
    use_adapter=False,
    adapter_hidden=1024,
)


class DoudizhuNet(nn.Module):
    def __init__(self, n_actions: int, cfg: dict | None = None):
        super().__init__()
        c = dict(DEFAULT_CFG)
        if cfg:
            c.update(cfg)
        self.cfg = c
        self.n_actions = int(n_actions)
        sh = c["state_hidden"]
        lh = c["lstm_hidden"]
        ah = c["action_hidden"]
        qh = c["q_hidden"]

        self.state_mlp = nn.Sequential(
            nn.Linear(c["state_dim"], sh), nn.ReLU(),
            nn.Linear(sh, sh), nn.ReLU(),
        )
        self.lstm = nn.LSTM(c["hist_dim"], lh, num_layers=1, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(sh + lh, sh), nn.ReLU())
        self.adapter = None
        if c.get("use_adapter"):
            self.adapter = nn.Sequential(
                nn.Linear(sh, c.get("adapter_hidden", 1024)), nn.GELU(),
                nn.Linear(c.get("adapter_hidden", 1024), sh),
            )
        self.action_emb = nn.Embedding(self.n_actions, ah)
        self.action_mlp = nn.Sequential(nn.Linear(c["action_feat_dim"], ah), nn.ReLU())
        self.q_head = nn.Sequential(
            nn.Linear(sh + ah, qh), nn.ReLU(),
            nn.Linear(qh, qh // 2), nn.ReLU(),
            nn.Linear(qh // 2, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.q_head[-1].weight, gain=0.01)
        for name, p in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        # 遗忘门偏置 1（PyTorch 门顺序 i, f, g, o）
        with torch.no_grad():
            for name, p in self.lstm.named_parameters():
                if "bias" in name:
                    h = p.shape[0] // 4
                    p[h:2 * h].fill_(1.0)
        nn.init.normal_(self.action_emb.weight, std=0.02)
        if self.adapter is not None:
            # 残差 adapter 最后一层零初始化：初始状态与旧模型完全等价
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

    def encode(self, state: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """state [B, STATE_DIM], hist [B, HIST, HIST_DIM] -> [B, sh]"""
        h = self.state_mlp(state)
        out, _ = self.lstm(hist)
        z = self.fuse(torch.cat([h, out[:, -1]], dim=-1))
        if self.adapter is not None:
            z = z + self.adapter(z)
        return z

    def q_from_z(self, z: torch.Tensor, action_ids: torch.Tensor,
                 action_feat: torch.Tensor) -> torch.Tensor:
        e = self.action_emb(action_ids) + self.action_mlp(action_feat)
        return self.q_head(torch.cat([z, e], dim=-1)).squeeze(-1)

    def q(self, state, hist, action_ids, action_feat):
        return self.q_from_z(self.encode(state, hist), action_ids, action_feat)


def build_model(n_actions: int, cfg: dict | None = None) -> DoudizhuNet:
    return DoudizhuNet(n_actions, cfg)


def save_checkpoint(path: str, model: DoudizhuNet, optimizer=None, meta: dict | None = None):
    payload = {
        "model": model.state_dict(),
        "cfg": model.cfg,
        "n_actions": model.n_actions,
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location="cpu", with_optimizer=False):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model = DoudizhuNet(payload["n_actions"], payload.get("cfg"))
    # 兼容旧 checkpoint（无 adapter）和新增 adapter 的模型
    model.load_state_dict(payload["model"], strict=False)
    model.eval()
    return model, payload
