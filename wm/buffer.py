"""Sequence replay. The unit is a trajectory chunk, not a transition.

A world model is trained on sequences because its whole job is to carry state
across time. Sampling independent transitions, which is what a model free
off-policy agent does, would make the recurrence untrainable.
"""
from __future__ import annotations

import torch


class SequenceBuffer:
    def __init__(self, capacity: int, horizon: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = torch.zeros(capacity, horizon + 1, obs_dim)
        self.act = torch.zeros(capacity, horizon, act_dim)
        self.rew = torch.zeros(capacity, horizon)
        self.n = 0
        self.ptr = 0

    def add(self, obs, act, rew):
        b = obs.shape[0]
        for i in range(b):
            self.obs[self.ptr] = obs[i]
            self.act[self.ptr] = act[i]
            self.rew[self.ptr] = rew[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.n = min(self.n + 1, self.capacity)

    def sample(self, batch: int, generator=None):
        idx = torch.randint(0, self.n, (batch,), generator=generator)
        return self.obs[idx], self.act[idx], self.rew[idx]
