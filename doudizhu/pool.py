"""对手池 / 简化联赛：保存历史快照，自对弈时随机让一个座位使用旧策略。"""
from __future__ import annotations

import torch

from .model import build_model


class OpponentPool:
    def __init__(self, model, n: int = 3, device="cuda"):
        self.n = int(n)
        self.device = torch.device(device)
        self.models = []
        for _ in range(self.n):
            m = build_model(model.n_actions, model.cfg).to(self.device)
            m.load_state_dict(model.state_dict())
            m.eval()
            self.models.append(m)
        self.ptr = 0
        self.updates = 0

    def add(self, model):
        m = self.models[self.ptr % self.n]
        m.load_state_dict(model.state_dict())
        m.eval()
        self.ptr += 1
        self.updates += 1
