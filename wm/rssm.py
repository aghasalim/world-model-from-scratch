"""Recurrent state space model.

The latent state is split in two, which is the central design of the RSSM:

    h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])      deterministic path
    z_t ~ q(z | h_t, o_t)                        posterior, sees the observation
    z_t ~ p(z | h_t)                             prior, does not

The deterministic path carries information forward reliably; a purely stochastic
state would have to re-encode everything at every step and loses long range
information to sampling noise. The stochastic part exists because the world is
not deterministic and a point estimate cannot represent "I do not know which way
it is swinging".

Training maximises a variational bound: predict the observation and reward from
the posterior, and keep the prior close to the posterior so that rolling forward
without observations still works. **The prior matching term is the entire reason
imagination is possible**, because imagination is exactly running the prior with
no observations to correct it.

KL balancing, from DreamerV2: the KL is computed twice with a stop gradient on
each side and mixed,

    KL = alpha * KL(sg(q) || p) + (1 - alpha) * KL(q || sg(p))

with alpha around 0.8. Without it the model prefers to make the posterior lazy,
collapsing q toward the uninformative prior rather than making the prior sharp,
because that is the cheaper way to reduce the same term. That failure is
measured in this repo rather than asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class RSSMConfig:
    deter: int = 64
    stoch: int = 16
    hidden: int = 128
    obs_dim: int = 2
    act_dim: int = 1
    kl_scale: float = 1.0
    kl_balance: float = 0.8
    free_nats: float = 0.1
    min_std: float = 0.1


def _dist(params, min_std: float):
    mean, std = params.chunk(2, dim=-1)
    return torch.distributions.Normal(mean, nn.functional.softplus(std) + min_std)


class RSSM(nn.Module):
    def __init__(self, cfg: RSSMConfig, decode_obs: bool = True, contrastive: bool = False):
        super().__init__()
        self.cfg = cfg
        self.decode_obs = decode_obs
        self.contrastive = contrastive
        d, s, h = cfg.deter, cfg.stoch, cfg.hidden

        self.embed = nn.Sequential(nn.Linear(cfg.obs_dim, h), nn.ELU(), nn.Linear(h, h), nn.ELU())
        self.cell = nn.GRUCell(s + cfg.act_dim, d)
        self.prior_net = nn.Sequential(nn.Linear(d, h), nn.ELU(), nn.Linear(h, 2 * s))
        self.post_net = nn.Sequential(nn.Linear(d + h, h), nn.ELU(), nn.Linear(h, 2 * s))

        feat = d + s
        self.reward_head = nn.Sequential(nn.Linear(feat, h), nn.ELU(), nn.Linear(h, 1))
        if decode_obs:
            self.decoder = nn.Sequential(nn.Linear(feat, h), nn.ELU(), nn.Linear(h, cfg.obs_dim))
        if contrastive:
            self.project = nn.Sequential(nn.Linear(feat, h), nn.ELU(), nn.Linear(h, h))

    def initial(self, n: int):
        return (torch.zeros(n, self.cfg.deter), torch.zeros(n, self.cfg.stoch))

    def observe(self, obs, act):
        """Filter a sequence. obs (B,T+1,O), act (B,T,A). Returns posteriors and priors."""
        b, t, _ = act.shape
        h, z = self.initial(b)
        embeds = self.embed(obs)
        posts, priors, feats = [], [], []
        for i in range(t):
            h = self.cell(torch.cat([z, act[:, i]], -1), h)
            prior = _dist(self.prior_net(h), self.cfg.min_std)
            post = _dist(self.post_net(torch.cat([h, embeds[:, i + 1]], -1)), self.cfg.min_std)
            z = post.rsample()
            posts.append(post)
            priors.append(prior)
            feats.append(torch.cat([h, z], -1))
        return posts, priors, torch.stack(feats, 1), (h, z)

    def imagine(self, state, policy, horizon: int):
        """Roll the PRIOR forward with no observations. This is imagination."""
        h, z = state
        feats, actions = [], []
        for _ in range(horizon):
            feat = torch.cat([h, z], -1)
            a = policy(feat)
            h = self.cell(torch.cat([z, a], -1), h)
            z = _dist(self.prior_net(h), self.cfg.min_std).rsample()
            feats.append(torch.cat([h, z], -1))
            actions.append(a)
        return torch.stack(feats, 1), torch.stack(actions, 1)

    def kl_loss(self, posts, priors):
        total = 0.0
        a = self.cfg.kl_balance
        for q, p in zip(posts, priors):
            q_sg = torch.distributions.Normal(q.mean.detach(), q.stddev.detach())
            p_sg = torch.distributions.Normal(p.mean.detach(), p.stddev.detach())
            lhs = torch.distributions.kl_divergence(q_sg, p).sum(-1)
            rhs = torch.distributions.kl_divergence(q, p_sg).sum(-1)
            kl = a * lhs + (1 - a) * rhs
            total = total + kl.clamp(min=self.cfg.free_nats).mean()
        return total / len(posts)

    def loss(self, obs, act, rew):
        posts, priors, feats, _ = self.observe(obs, act)
        parts = {}
        pred_r = self.reward_head(feats).squeeze(-1)
        parts["reward"] = ((pred_r - rew) ** 2).mean()

        if self.decode_obs:
            parts["recon"] = ((self.decoder(feats) - obs[:, 1:]) ** 2).mean()
        if self.contrastive:
            # InfoNCE between the latent feature and the embedding of the
            # observation it should correspond to, over the batch.
            q = nn.functional.normalize(self.project(feats), dim=-1)
            k = nn.functional.normalize(self.embed(obs[:, 1:]), dim=-1)
            b, t, _d = q.shape
            logits = torch.einsum("btd,ctd->btc", q, k) / 0.1
            target = torch.arange(b).unsqueeze(1).expand(b, t)
            parts["contrastive"] = nn.functional.cross_entropy(
                logits.reshape(b * t, b), target.reshape(b * t))

        parts["kl"] = self.cfg.kl_scale * self.kl_loss(posts, priors)
        total = sum(parts.values())
        return total, {k: v.detach().item() for k, v in parts.items()}, feats
