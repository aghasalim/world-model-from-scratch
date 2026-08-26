"""Sample efficiency, the representation ablation, and open loop prediction.

Three questions:

  1. Does learning in imagination beat a model free baseline per environment
     step? The baseline is given the same recurrent architecture so the
     comparison isolates imagination rather than memory.
  2. Does the world model need to reconstruct observations? Dreamer does and
     pays capacity for it. MuZero style models predict only reward and value.
     Contrastive sits between. This is the part that is not a reimplementation.
  3. How far can the model roll forward before its predictions diverge?

    .venv/bin/python -m experiments.main
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch

from wm.agent import Actor, Critic, imagine_and_learn
from wm.buffer import SequenceBuffer
from wm.envs import PendulumPOMDP, random_policy
from wm.modelfree import RecurrentPolicy, collect, update
from wm.rssm import RSSM, RSSMConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODES = {
    "recon (Dreamer style)": {"decode_obs": True, "contrastive": False},
    "no-recon (MuZero style)": {"decode_obs": False, "contrastive": False},
    "contrastive": {"decode_obs": False, "contrastive": True},
}


@torch.no_grad()
def evaluate(env_n, horizon, rssm, actor, seed):
    """Return of the learned policy, acting through the filtered latent state."""
    env = PendulumPOMDP(env_n, horizon=horizon, seed=seed)
    obs = env.reset()
    h, z = rssm.initial(env_n)
    total = torch.zeros(env_n)
    a = torch.zeros(env_n, 1)
    for _ in range(horizon):
        h = rssm.cell(torch.cat([z, a], -1), h)
        from wm.rssm import _dist
        post = _dist(rssm.post_net(torch.cat([h, rssm.embed(obs)], -1)), rssm.cfg.min_std)
        z = post.mean
        a = actor(torch.cat([h, z], -1))
        obs, r, _ = env.step(a)
        total += r
    return total.mean().item()


@torch.no_grad()
def open_loop_error(rssm, env_n, horizon, context, seed):
    """Filter `context` steps, then predict forward with no observations."""
    env = PendulumPOMDP(env_n, horizon=horizon, seed=seed)
    g = torch.Generator().manual_seed(seed + 1)
    obs, act, rew = env.rollout(lambda o, s: random_policy(o, s, g))
    _, _, _, state = rssm.observe(obs[:, :context + 1], act[:, :context])
    h, z = state
    errs = []
    for k in range(horizon - context):
        a = act[:, context + k]
        h = rssm.cell(torch.cat([z, a], -1), h)
        from wm.rssm import _dist
        z = _dist(rssm.prior_net(h), rssm.cfg.min_std).mean
        feat = torch.cat([h, z], -1)
        pred_r = rssm.reward_head(feat).squeeze(-1)
        errs.append({"step": k + 1,
                     "reward_mae": (pred_r - rew[:, context + k]).abs().mean().item()})
    return errs


def train_world_model(mode, kwargs, args, seed):
    torch.manual_seed(seed)
    cfg = RSSMConfig(kl_balance=args.kl_balance)
    rssm = RSSM(cfg, **kwargs)
    actor, critic = Actor(cfg.deter + cfg.stoch, 1), Critic(cfg.deter + cfg.stoch)
    import copy as _copy
    target_critic = _copy.deepcopy(critic)
    for p_ in target_critic.parameters():
        p_.requires_grad_(False)
    wm_opt = torch.optim.Adam(rssm.parameters(), lr=args.wm_lr)
    a_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    c_opt = torch.optim.Adam(critic.parameters(), lr=args.actor_lr)

    buf = SequenceBuffer(args.buffer, args.horizon, 2, 1)
    env = PendulumPOMDP(args.envs, horizon=args.horizon, seed=seed)
    g = torch.Generator().manual_seed(seed + 7)
    env_steps, curve = 0, []
    t0 = time.perf_counter()

    for it in range(args.iters):
        # ---- collect with the current policy (random for the first round)
        if it == 0:
            obs, act, rew = env.rollout(lambda o, s: random_policy(o, s, g))
        else:
            def pol(o, s):
                # state is (h, z, previous action); the previous action is needed
                # because the GRU transition consumes it.
                if s is None:
                    h, z = rssm.initial(o.shape[0])
                    a_prev = torch.zeros(o.shape[0], 1)
                else:
                    h, z, a_prev = s
                h = rssm.cell(torch.cat([z, a_prev], -1), h)
                from wm.rssm import _dist
                post = _dist(rssm.post_net(torch.cat([h, rssm.embed(o)], -1)), rssm.cfg.min_std)
                z = post.mean
                a = actor(torch.cat([h, z], -1))
                a = (a + args.explore * torch.randn(a.shape, generator=g)).clamp(-1, 1)
                return a, (h, z, a)
            with torch.no_grad():
                obs, act, rew = env.rollout(pol)
        buf.add(obs, act, rew)
        env_steps += args.envs * args.horizon

        # ---- fit the model, then improve the policy purely in imagination
        for _ in range(args.wm_updates):
            o, a, r = buf.sample(args.batch, g)
            loss, parts, _ = rssm.loss(o, a, r)
            wm_opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
            wm_opt.step()

        for _ in range(args.actor_updates):
            o, a, r = buf.sample(args.batch, g)
            with torch.no_grad():
                _, _, _, state = rssm.observe(o, a)
            imagine_and_learn(rssm, actor, critic, (state[0].detach(), state[1].detach()),
                              args.imag_horizon, a_opt, c_opt,
                              target_critic=target_critic)

        if it % args.eval_every == 0 or it == args.iters - 1:
            ret = evaluate(64, args.horizon, rssm, actor, seed + 100)
            curve.append({"mode": mode, "seed": seed, "iter": it, "env_steps": env_steps,
                          "return": ret, **parts})
    return rssm, actor, curve, time.perf_counter() - t0, env_steps


def train_model_free(args, seed):
    torch.manual_seed(seed)
    policy = RecurrentPolicy(2, 1)
    opt = torch.optim.Adam(policy.parameters(), lr=args.mf_lr)
    curve, env_steps = [], 0
    t0 = time.perf_counter()
    for it in range(args.mf_iters):
        env = PendulumPOMDP(args.envs, horizon=args.horizon, seed=seed * 31 + it)
        lp, r, v, e = collect(env, policy)
        update(policy, opt, lp, r, v, e)
        env_steps += args.envs * args.horizon
        if it % args.mf_eval_every == 0 or it == args.mf_iters - 1:
            curve.append({"mode": "model-free (recurrent PG)", "seed": seed, "iter": it,
                          "env_steps": env_steps, "return": r.sum(dim=1).mean().item()})
    return curve, time.perf_counter() - t0, env_steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--buffer", type=int, default=400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--wm-updates", type=int, default=25)
    ap.add_argument("--actor-updates", type=int, default=8)
    ap.add_argument("--imag-horizon", type=int, default=15)
    ap.add_argument("--wm-lr", type=float, default=3e-4)
    ap.add_argument("--actor-lr", type=float, default=8e-5)
    ap.add_argument("--mf-lr", type=float, default=3e-4)
    ap.add_argument("--mf-iters", type=int, default=400)
    ap.add_argument("--mf-eval-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--explore", type=float, default=0.3)
    ap.add_argument("--kl-balance", type=float, default=0.8)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    curves, openloop, summary = [], [], []
    started = time.perf_counter()

    for seed in args.seeds:
        for mode, kw in MODES.items():
            rssm, _actor, curve, wall, steps = train_world_model(mode, kw, args, seed)
            curves += curve
            final = curve[-1]["return"]
            summary.append({"mode": mode, "seed": seed, "final_return": final,
                            "env_steps": steps, "wall_s": wall,
                            "params": sum(p.numel() for p in rssm.parameters())})
            for e in open_loop_error(rssm, 128, args.horizon, 20, seed + 500):
                openloop.append({"mode": mode, "seed": seed, **e})
            print(f"  seed {seed}  {mode:26} return {final:+7.2f}  "
                  f"{steps} env steps  {wall:5.0f}s")

        curve, wall, steps = train_model_free(args, seed)
        curves += curve
        summary.append({"mode": "model-free (recurrent PG)", "seed": seed,
                        "final_return": curve[-1]["return"], "env_steps": steps,
                        "wall_s": wall, "params": 0})
        print(f"  seed {seed}  {'model-free (recurrent PG)':26} return "
              f"{curve[-1]['return']:+7.2f}  {steps} env steps  {wall:5.0f}s")

    for fname, rows in (("learning-curves.csv", curves), ("open-loop.csv", openloop),
                        ("summary.csv", summary)):
        p = RESULTS / fname
        with p.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {p.relative_to(ROOT)} ({len(rows)} rows)")
    (RESULTS / "run-meta.json").write_text(json.dumps({
        **vars(args), "wall_clock_s": time.perf_counter() - started,
        "torch": torch.__version__, "device": "cpu"}, indent=1))
    print(f"total {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
