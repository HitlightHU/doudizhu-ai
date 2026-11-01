"""经验回放：环形缓冲区，存储 float16 特征。"""
from __future__ import annotations

import numpy as np

from .engine import HIST, HIST_DIM, STATE_DIM

HIST_FLAT = HIST * HIST_DIM


class ReplayBuffer:
    def __init__(self, capacity: int = 400_000, seed: int = 0):
        self.capacity = int(capacity)
        self.states = np.zeros((self.capacity, STATE_DIM), dtype=np.float16)
        self.hists = np.zeros((self.capacity, HIST_FLAT), dtype=np.float16)
        self.actions = np.zeros(self.capacity, dtype=np.int32)
        self.rewards = np.zeros(self.capacity, dtype=np.float16)
        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, states: np.ndarray, hists: np.ndarray, actions: np.ndarray,
            rewards: np.ndarray):
        n = len(actions)
        if n == 0:
            return
        if n > self.capacity:
            states = states[-self.capacity:]
            hists = hists[-self.capacity:]
            actions = actions[-self.capacity:]
            rewards = rewards[-self.capacity:]
            n = self.capacity
        end = self.ptr + n
        if end <= self.capacity:
            self.states[self.ptr:end] = states
            self.hists[self.ptr:end] = hists
            self.actions[self.ptr:end] = actions
            self.rewards[self.ptr:end] = rewards
        else:
            first = self.capacity - self.ptr
            self.states[self.ptr:] = states[:first]
            self.hists[self.ptr:] = hists[:first]
            self.actions[self.ptr:] = actions[:first]
            self.rewards[self.ptr:] = rewards[:first]
            rest = n - first
            self.states[:rest] = states[first:]
            self.hists[:rest] = hists[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
        self.ptr = end % self.capacity
        self.size = min(self.capacity, self.size + n)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (self.states[idx], self.hists[idx], self.actions[idx],
                self.rewards[idx])
