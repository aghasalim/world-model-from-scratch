"""Export reference weights and rollouts so other languages can recompute them.

The dynamics step is the one piece of this repository that only exists in
PyTorch. Every open loop number in the README comes out of

    h = GRUCell([z, a], h)
    z = mean of prior_net(h)
    r = reward_head([h, z])

run forward with no observations, and nothing checked that loop against a second
implementation. This writes a seeded reference so verify/rssm_step.c can.

The weights are a seeded initialisation rather than a trained checkpoint. No
checkpoint is committed, and what is being verified here is the arithmetic of
the step, which does not depend on the values. The starting state is the one
PyTorch produces after filtering twenty observations, so the C program is handed
the same problem the published measurement solves.

The pendulum trajectory is exported for the same reason: verify/pendulum in Rust
reruns the environment millions of times, and that only means anything if its
integrator is the same one wm/envs.py uses.

    python verify/export_golden.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wm.envs import PendulumPOMDP, random_policy
from wm.rssm import RSSM, RSSMConfig, _dist

OUT = ROOT / "verify" / "golden"
CONTEXT = 20
STEPS = 40
SEED = 0


def dump(fh, name, t):
    t = t.detach()
    if t.dim() == 1:
        t = t.unsqueeze(0)
    fh.write(f"{name} {t.shape[0]} {t.shape[1]}\n")
    for row in t:
        fh.write(" ".join(f"{v:.9e}" for v in row.tolist()) + "\n")


@torch.no_grad()
def export_rssm():
    torch.manual_seed(SEED)
    cfg = RSSMConfig()
    rssm = RSSM(cfg, decode_obs=True)
    rssm.eval()

    env = PendulumPOMDP(4, horizon=CONTEXT + STEPS, seed=SEED + 500)
    g = torch.Generator().manual_seed(SEED + 1)
    obs, act, _rew = env.rollout(lambda o, s: random_policy(o, s, g))

    # Filtering uses the posterior, which samples. Take the mean instead so the
    # starting state is reproducible; the loop being verified starts after it.
    h, z = rssm.initial(obs.shape[0])
    embeds = rssm.embed(obs)
    for i in range(CONTEXT):
        h = rssm.cell(torch.cat([z, act[:, i]], -1), h)
        z = _dist(rssm.post_net(torch.cat([h, embeds[:, i + 1]], -1)), cfg.min_std).mean

    with (OUT / "rssm_weights.txt").open("w") as fh:
        fh.write(f"# deter {cfg.deter} stoch {cfg.stoch} hidden {cfg.hidden} "
                 f"act {cfg.act_dim} min_std {cfg.min_std}\n")
        dump(fh, "cell.weight_ih", rssm.cell.weight_ih)
        dump(fh, "cell.weight_hh", rssm.cell.weight_hh)
        dump(fh, "cell.bias_ih", rssm.cell.bias_ih)
        dump(fh, "cell.bias_hh", rssm.cell.bias_hh)
        dump(fh, "prior.0.weight", rssm.prior_net[0].weight)
        dump(fh, "prior.0.bias", rssm.prior_net[0].bias)
        dump(fh, "prior.2.weight", rssm.prior_net[2].weight)
        dump(fh, "prior.2.bias", rssm.prior_net[2].bias)
        dump(fh, "reward.0.weight", rssm.reward_head[0].weight)
        dump(fh, "reward.0.bias", rssm.reward_head[0].bias)
        dump(fh, "reward.2.weight", rssm.reward_head[2].weight)
        dump(fh, "reward.2.bias", rssm.reward_head[2].bias)

    with (OUT / "rssm_rollout.txt").open("w") as fh:
        fh.write(f"# batch {obs.shape[0]} steps {STEPS}\n")
        dump(fh, "h0", h)
        dump(fh, "z0", z)
        dump(fh, "actions", act[:, CONTEXT:CONTEXT + STEPS, 0])
        hs, zs, rs = [], [], []
        for k in range(STEPS):
            a = act[:, CONTEXT + k]
            h = rssm.cell(torch.cat([z, a], -1), h)
            z = _dist(rssm.prior_net(h), cfg.min_std).mean
            hs.append(h.clone())
            zs.append(z.clone())
            rs.append(rssm.reward_head(torch.cat([h, z], -1)).squeeze(-1).clone())
        dump(fh, "h", torch.stack(hs, 1).reshape(obs.shape[0], -1))
        dump(fh, "z", torch.stack(zs, 1).reshape(obs.shape[0], -1))
        dump(fh, "reward", torch.stack(rs, 1))
    print(f"wrote {OUT / 'rssm_weights.txt'} and {OUT / 'rssm_rollout.txt'}")


@torch.no_grad()
def export_pendulum():
    n, steps = 8, 60
    env = PendulumPOMDP(n, horizon=steps, seed=SEED + 500)
    g = torch.Generator().manual_seed(SEED + 1)
    theta0, thetadot0 = env.theta.clone(), env.theta_dot.clone()
    acts, ths, thds, rews = [], [], [], []
    for _ in range(steps):
        a = (torch.rand(n, 1, generator=g) * 2 - 1)
        _obs, r, _d = env.step(a)
        acts.append(a.squeeze(-1))
        ths.append(env.theta.clone())
        thds.append(env.theta_dot.clone())
        rews.append(r.clone())
    with (OUT / "pendulum.txt").open("w") as fh:
        fh.write(f"# batch {n} steps {steps}\n")
        dump(fh, "theta0", theta0)
        dump(fh, "theta_dot0", thetadot0)
        dump(fh, "actions", torch.stack(acts, 1))
        dump(fh, "theta", torch.stack(ths, 1))
        dump(fh, "theta_dot", torch.stack(thds, 1))
        dump(fh, "reward", torch.stack(rews, 1))
    print(f"wrote {OUT / 'pendulum.txt'}")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    export_rssm()
    export_pendulum()
