"""Actor and critic, trained entirely inside the world model.

No environment steps are used here at all. The agent starts from states the
world model filtered out of real data, imagines `horizon` steps forward using
only the prior, and learns from the imagined rewards. That is the sample
efficiency claim: real experience is spent learning the model, and the policy is
improved thousands of times per real step.

Returns use the lambda formulation from DreamerV2, which interpolates between a
one step TD target and the full imagined return. Short horizons make the
estimate low variance and model-biased; long ones make it high variance and
compounding-error-prone. Lambda picks a point on that line.

The actor is trained by backpropagating through the imagined trajectory, which
is possible because every step of the RSSM prior is differentiable. That is a
real advantage over model free policy gradients, and also the reason model error
hurts here in a way it does not hurt a model free method.
"""
from __future__ import annotations

import torch
from torch import nn


class Actor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, act_dim), nn.Tanh())

    def forward(self, feat):
        return self.net(feat)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feat_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU(),
                                 nn.Linear(hidden, 1))

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


def lambda_return(rewards, values, gamma=0.99, lam=0.95):
    """DreamerV2 style lambda returns over an imagined trajectory."""
    T = rewards.shape[1]
    out = torch.zeros_like(rewards)
    last = values[:, -1]
    for t in reversed(range(T)):
        boot = values[:, t + 1] if t + 1 < T else values[:, -1]
        last = rewards[:, t] + gamma * ((1 - lam) * boot + lam * last)
        out[:, t] = last
    return out


def imagine_and_learn(rssm, actor, critic, start_state, horizon,
                      actor_opt, critic_opt, gamma=0.99, lam=0.95,
                      target_critic=None, target_tau=0.02):
    """One actor-critic update from imagined rollouts. No environment steps.

    `target_critic` is a slowly updated copy used to compute the bootstrap
    values. Without it the critic regresses toward targets built from its own
    current output, and on this task that showed up as a policy that reached a
    return of about -30 and then fell back to -39: not divergence, just an
    oscillation that never settled. DreamerV2 uses a target network for exactly
    this and omitting it was my own shortcut.
    """
    feats, _ = rssm.imagine(start_state, actor, horizon)
    rewards = rssm.reward_head(feats).squeeze(-1)
    boot = (target_critic or critic)(feats)
    returns = lambda_return(rewards, boot, gamma, lam)

    actor_loss = -returns.mean()
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward(retain_graph=True)
    nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
    actor_opt.step()

    critic_loss = ((critic(feats.detach()) - returns.detach()) ** 2).mean()
    critic_opt.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
    critic_opt.step()

    if target_critic is not None:
        with torch.no_grad():
            for tp, p in zip(target_critic.parameters(), critic.parameters()):
                tp.mul_(1 - target_tau).add_(p, alpha=target_tau)

    return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(),
            "imagined_return": returns[:, 0].mean().item()}
