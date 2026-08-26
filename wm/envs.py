"""A partially observable pendulum, implemented directly.

Classic pendulum swing-up dynamics, but the agent observes only cos(theta) and
sin(theta). Angular velocity is hidden.

That choice is the point of the repo. With velocity observed the task is a plain
MDP and a feedforward policy solves it, so nothing would test whether the model
has learned to carry state. Hiding velocity makes it a POMDP: the only way to
know which way the pendulum is moving is to remember where it was, which is
exactly what the recurrent state of an RSSM is for.

Batched and written in torch so a few hundred environments step at once. Wall
clock in this repo is dominated by environment stepping, same as the real thing.
"""
from __future__ import annotations

import math

import torch

MAX_SPEED = 8.0
MAX_TORQUE = 2.0
DT = 0.05
GRAVITY = 10.0
MASS = 1.0
LENGTH = 1.0


class PendulumPOMDP:
    """Vectorised pendulum. State is (theta, theta_dot); observation hides theta_dot."""

    obs_dim = 2
    act_dim = 1

    def __init__(self, n: int, horizon: int = 100, seed: int = 0, observe_velocity: bool = False):
        self.n, self.horizon = n, horizon
        self.observe_velocity = observe_velocity
        if observe_velocity:
            self.obs_dim = 3
        self.g = torch.Generator().manual_seed(seed)
        self.theta = torch.zeros(n)
        self.theta_dot = torch.zeros(n)
        self.t = 0
        self.reset()

    def reset(self) -> torch.Tensor:
        self.theta = (torch.rand(self.n, generator=self.g) * 2 - 1) * math.pi
        self.theta_dot = (torch.rand(self.n, generator=self.g) * 2 - 1) * 1.0
        self.t = 0
        return self.observe()

    def observe(self) -> torch.Tensor:
        parts = [self.theta.cos(), self.theta.sin()]
        if self.observe_velocity:
            parts.append(self.theta_dot / MAX_SPEED)
        return torch.stack(parts, dim=-1)

    @staticmethod
    def angle_normalize(x):
        return ((x + math.pi) % (2 * math.pi)) - math.pi

    def step(self, action: torch.Tensor):
        """action in [-1, 1], shape (n, 1). Returns (obs, reward, done)."""
        u = action.squeeze(-1).clamp(-1.0, 1.0) * MAX_TORQUE
        th, thd = self.theta, self.theta_dot

        cost = self.angle_normalize(th) ** 2 + 0.1 * thd ** 2 + 0.001 * u ** 2
        reward = -cost / 10.0            # scaled into roughly [-1.7, 0]

        new_thd = thd + (3 * GRAVITY / (2 * LENGTH) * th.sin()
                         + 3.0 / (MASS * LENGTH ** 2) * u) * DT
        new_thd = new_thd.clamp(-MAX_SPEED, MAX_SPEED)
        self.theta = th + new_thd * DT
        self.theta_dot = new_thd
        self.t += 1
        done = torch.full((self.n,), self.t >= self.horizon, dtype=torch.bool)
        return self.observe(), reward, done

    @torch.no_grad()
    def rollout(self, policy, steps: int | None = None):
        """Collect a batch of sequences. `policy(obs, state) -> (action, state)`."""
        steps = steps or self.horizon
        obs = self.reset()
        state = None
        O, A, R = [obs], [], []
        for _ in range(steps):
            action, state = policy(obs, state)
            obs, reward, _ = self.step(action)
            O.append(obs)
            A.append(action)
            R.append(reward)
        return (torch.stack(O, 1), torch.stack(A, 1), torch.stack(R, 1))


def random_policy(obs, state, generator=None):
    n = obs.shape[0]
    return (torch.rand(n, 1, generator=generator) * 2 - 1), state
