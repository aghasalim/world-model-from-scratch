"""A model free baseline, for the sample efficiency comparison.

Deliberately given the same recurrent architecture as the world model agent, so
the comparison isolates *learning in imagination* rather than accidentally
comparing a recurrent policy against a feedforward one on a POMDP, which the
recurrent one would win for a reason that has nothing to do with world models.

This is a straightforward recurrent policy gradient with a learned baseline,
trained on real environment returns only.
"""
from __future__ import annotations

import torch
from torch import nn


class RecurrentPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.cell = nn.GRUCell(obs_dim, hidden)
        self.mu = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(),
                                nn.Linear(hidden, act_dim), nn.Tanh())
        self.value = nn.Sequential(nn.Linear(hidden, hidden), nn.ELU(), nn.Linear(hidden, 1))
        self.hidden = hidden
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))

    def initial(self, n):
        return torch.zeros(n, self.hidden)

    def forward(self, obs, h):
        h = self.cell(obs, h)
        return self.mu(h), self.value(h).squeeze(-1), h


def collect(env, policy, generator=None):
    obs = env.reset()
    h = policy.initial(env.n)
    logps, rewards, values, ents = [], [], [], []
    for _ in range(env.horizon):
        mu, v, h = policy(obs, h)
        dist = torch.distributions.Normal(mu, policy.log_std.exp())
        a = dist.rsample()
        logps.append(dist.log_prob(a).sum(-1))
        ents.append(dist.entropy().sum(-1))
        values.append(v)
        obs, r, _ = env.step(a.detach())
        rewards.append(r)
    return (torch.stack(logps, 1), torch.stack(rewards, 1),
            torch.stack(values, 1), torch.stack(ents, 1))


def update(policy, opt, logps, rewards, values, ents, gamma=0.99, ent_coef=1e-3):
    returns = torch.zeros_like(rewards)
    run = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(rewards.shape[1])):
        run = rewards[:, t] + gamma * run
        returns[:, t] = run
    adv = (returns - values).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss = -(logps * adv).mean() + ((values - returns.detach()) ** 2).mean() \
        - ent_coef * ents.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    opt.step()
    return {"loss": loss.item(), "return": returns[:, 0].mean().item()}
