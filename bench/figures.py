"""Figures from committed CSVs, plus one animation that re-simulates.

The three static figures only read results/*.csv. They never re-run an
experiment and never write one.

The animation is the exception. Showing a dream drift away from reality needs a
trajectory, and a trajectory is not something a summary CSV can hold, so this
retrains the reconstruction world model at seed 0 with the hyperparameters
recorded in results/run-meta.json. That costs about a minute on a laptop CPU.
Before drawing anything it recomputes the open loop reward error and compares it
against the committed open-loop.csv. If those disagree the animation is skipped
rather than written, because then it would no longer be a picture of the run the
rest of the repo reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

from bench.style import PALETTE, titled

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

RECON = "recon (Dreamer style)"
NORECON = "no-recon (MuZero style)"
# Nothing in the README ties a meaning to any of these colours, so they come
# straight from the shared palette. Recon takes the first slot because it is the
# arm the ablation is about.
C = {RECON: PALETTE[0], NORECON: PALETTE[3],
     "contrastive": PALETTE[4], "model-free (recurrent PG)": PALETTE[5]}

DT = 0.05          # seconds per step, from wm/envs.py
CONTEXT = 20       # filtered steps before the observations are cut off
RANDOM_RETURN = -36.9   # measured over 200,000 episodes, see verify/pendulum


def _seeds(ax, frame, x, y, colour, label, marker=None, window=1, alpha=0.25):
    """Median across seeds in bold, with each seed drawn faintly behind it.

    Three seeds is too few to hide behind a smoothed line, so the raw runs stay
    visible and the reader can see for themselves how wide the spread is.
    `window` puts a short rolling median on the bold line, for the curves that
    are sampled often enough to bounce between neighbouring evaluations.
    """
    for _, s in frame.groupby("seed"):
        s = s.sort_values(x)
        ax.plot(s[x], s[y], color=colour, alpha=alpha, linewidth=0.9, zorder=1)
    g = frame.groupby(x)[y].median()
    if window > 1:
        g = g.rolling(window, center=True, min_periods=1).median()
    ax.plot(g.index, g.values, color=colour, label=label, marker=marker, zorder=3)


def fig_open_loop(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "open-loop.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.9))

    for mode, colour in C.items():
        s = t[t["mode"] == mode]
        if s.empty:
            continue
        _seeds(a, s, "step", "reward_mae", colour, mode)
        _seeds(b, s[s["step"] <= 10], "step", "reward_mae", colour, mode, marker="o")

    a.set_xlabel(f"open loop step k (one step is {DT:g} s of simulated time)")
    a.set_ylabel("reward prediction MAE (reward units)")
    titled(a, "Every model drifts once the observations stop",
           "bold line is the median of 3 seeds, thin lines are the seeds themselves")
    a.legend(loc="upper left")

    b.set_xlabel("open loop step k")
    b.set_ylabel("reward prediction MAE (reward units)")
    b.set_xticks(range(1, 11))
    titled(b, "Reconstruction only wins near the last observation",
           "at k=1 and k=5 the worst recon seed still beats the best no-recon seed")
    b.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_learning(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "learning-curves.csv")
    fig, ax = plt.subplots(figsize=(10.0, 5.2))

    ax.axhline(RANDOM_RETURN, color="#8a8a8a", linestyle=(0, (5, 4)), linewidth=1.2, zorder=0)
    for mode, colour in C.items():
        s = t[t["mode"] == mode]
        if not s.empty:
            # The model-free arm is evaluated three times as often and on the
            # exploring rollouts themselves, so it bounces from one evaluation
            # to the next. A three point rolling median takes that out without
            # touching the shape.
            _seeds(ax, s, "env_steps", "return", colour, mode,
                   window=3 if "model-free" in mode else 1,
                   alpha=0.15 if "model-free" in mode else 0.25)

    ax.set_xscale("log")
    ax.set_xlabel("environment steps (log scale)")
    ax.set_ylabel("episode return (sum of 60 step rewards)")
    ax.text(ax.get_xlim()[0] * 1.15, RANDOM_RETURN + 0.15, "random policy, about -36.9",
            ha="left", va="bottom", fontsize=9, color="#6a6a6a",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    titled(ax, "Nothing here reliably beats acting at random",
           "bold is the median over 3 seeds and the thin lines are the seeds "
           "themselves, model-free smoothed over 3 evaluations")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_model_fit(out: Path) -> Path:
    """The half that did work: the model itself gets better."""
    t = pd.read_csv(RESULTS / "learning-curves.csv")
    t = t[t["reward"].notna()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 4.7))

    for mode in (RECON, NORECON, "contrastive"):
        s = t[t["mode"] == mode]
        if s.empty:
            continue
        _seeds(a, s, "iter", "reward", C[mode], mode)
        _seeds(b, s, "iter", "kl", C[mode], mode)

    a.set_yscale("log")
    a.set_ylabel("reward prediction loss (MSE, log scale)")
    titled(a, "The world model does learn the dynamics",
           "reward prediction loss falls by more than an order of magnitude in every arm")
    a.legend(loc="upper right")

    b.set_ylabel("KL(posterior || prior), nats")
    titled(b, "The posterior stays informative",
           "KL rises instead of collapsing, which is what KL balancing is there to prevent")
    b.legend(loc="lower right")

    for ax in (a, b):
        ax.set_xlabel("training iteration")

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# --- animation --------------------------------------------------------------

def _documented_run():
    """The hyperparameters the committed results were produced with."""
    meta = json.loads((RESULTS / "run-meta.json").read_text())
    return argparse.Namespace(**{k: v for k, v in meta.items()
                                 if k not in ("wall_clock_s", "torch", "device")})


def _dream_and_reality(seed: int = 0, n: int = 128):
    """Retrain the recon model at `seed` and roll it forward with no observations.

    Mirrors experiments.main.open_loop_error step for step, and returns the
    per-episode detail that function throws away. The returned reward error must
    equal the committed open-loop.csv, which is checked by the caller.
    """
    import torch

    from experiments.main import MODES, train_world_model
    from wm.envs import PendulumPOMDP, random_policy
    from wm.rssm import _dist

    args = _documented_run()
    rssm, *_ = train_world_model(RECON, MODES[RECON], args, seed)

    eval_seed = seed + 500
    with torch.no_grad():
        env = PendulumPOMDP(n, horizon=args.horizon, seed=eval_seed)
        g = torch.Generator().manual_seed(eval_seed + 1)
        obs, act, rew = env.rollout(lambda o, s: random_policy(o, s, g))

        _, _, feats, state = rssm.observe(obs[:, :CONTEXT + 1], act[:, :CONTEXT])
        decoded = [rssm.decoder(feats)]
        pred_r = [rssm.reward_head(feats).squeeze(-1)]

        h, z = state
        mae = []
        for k in range(args.horizon - CONTEXT):
            h = rssm.cell(torch.cat([z, act[:, CONTEXT + k]], -1), h)
            z = _dist(rssm.prior_net(h), rssm.cfg.min_std).mean
            feat = torch.cat([h, z], -1)
            decoded.append(rssm.decoder(feat).unsqueeze(1))
            r = rssm.reward_head(feat).squeeze(-1)
            pred_r.append(r.unsqueeze(1))
            mae.append((r - rew[:, CONTEXT + k]).abs().mean().item())

    return {"obs": obs.numpy(), "rew": rew.numpy(),
            "decoded": torch.cat(decoded, 1).numpy(),
            "pred_r": torch.cat(pred_r, 1).numpy(),
            "mae": np.asarray(mae)}


def anim_dream_vs_real(out: Path, seed: int = 0, fps: int = 15) -> Path:
    roll = _dream_and_reality(seed=seed)

    ref = pd.read_csv(RESULTS / "open-loop.csv")
    ref = ref[(ref["mode"] == RECON) & (ref["seed"] == seed)].sort_values("step")
    drift = float(np.abs(roll["mae"] - ref["reward_mae"].to_numpy()).max())
    if drift > 1e-9:
        raise RuntimeError(
            f"re-simulated open loop error differs from open-loop.csv by {drift:.2e}, "
            "so this would not be the run the repo reports")

    # One of the 128 episodes has to be picked. Take the one whose open loop
    # reward error is the median, so the picture is neither the best case nor
    # the worst.
    err = np.abs(roll["pred_r"][:, CONTEXT:] - roll["rew"][:, CONTEXT:]).mean(1)
    i = int(np.argsort(err)[len(err) // 2])

    obs, dec = roll["obs"][i, 1:], roll["decoded"][i]
    rew, pred = roll["rew"][i], roll["pred_r"][i]
    steps = np.arange(1, len(rew) + 1)
    # theta = 0 is upright, and the observation is (cos theta, sin theta), so the
    # tip of the rod sits at (sin theta, cos theta).
    true_xy = np.stack([obs[:, 1], obs[:, 0]], -1)
    dream_xy = np.stack([dec[:, 1], dec[:, 0]], -1)
    cross = true_xy[:, 0] * dream_xy[:, 1] - true_xy[:, 1] * dream_xy[:, 0]
    gap = np.degrees(np.abs(np.arctan2(cross, (true_xy * dream_xy).sum(-1))))

    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                               gridspec_kw={"width_ratios": [1, 1.35]})

    ring = np.linspace(0, 2 * np.pi, 200)
    a.plot(np.cos(ring), np.sin(ring), color="#e2e2e2", linewidth=1.0, zorder=0)
    a.plot([0, 0], [0, 1.0], color="#d0d0d0", linestyle=(0, (2, 3)), linewidth=1.0, zorder=0)
    a.text(0, 1.12, "upright", ha="center", va="bottom", fontsize=8.5, color="#8a8a8a")
    a.plot([0], [0], marker="o", color="#8a8a8a", markersize=4, zorder=4)
    a.set_xlim(-1.45, 1.45)
    a.set_ylim(-1.45, 1.45)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)
    for sp in a.spines.values():
        sp.set_visible(False)
    titled(a, "Same start, same actions",
           "solid is the pendulum, dashed is what the model imagines")

    trail_t, = a.plot([], [], color=C[RECON], alpha=0.30, linewidth=1.4, zorder=1)
    trail_d, = a.plot([], [], color=C[NORECON], alpha=0.30, linewidth=1.4, zorder=1)
    rod_t, = a.plot([], [], color=C[RECON], linewidth=3.0, solid_capstyle="round",
                    marker="o", markevery=[1], markersize=9, zorder=5)
    rod_d, = a.plot([], [], color=C[NORECON], linewidth=2.2, linestyle=(0, (4, 2)),
                    marker="o", markevery=[1], markerfacecolor="white", markersize=8,
                    zorder=6)
    phase = a.text(0.5, 0.015, "", transform=a.transAxes, ha="center", va="bottom",
                   fontsize=10, color="#333333")

    b.axvspan(CONTEXT + 0.5, len(rew) + 0.5, color="#f4f4f4", zorder=0)
    b.axvline(CONTEXT + 0.5, color="#9a9a9a", linestyle=(0, (3, 3)), linewidth=1.1, zorder=1)
    b.set_xlim(0.5, len(rew) + 0.5)
    b.set_ylim(min(rew.min(), pred.min()) - 0.16, max(rew.max(), pred.max()) + 0.22)
    b.text(CONTEXT + 1.6, 0.955, "no observations from here on", transform=b.get_xaxis_transform(),
           fontsize=9, color="#6a6a6a", ha="left", va="top")
    b.set_xlabel(f"step (one step is {DT:g} s of simulated time)")
    b.set_ylabel("reward per step")
    titled(b, "The dream loses the timing, not the shape",
           "predicted reward against the reward the pendulum actually paid")
    line_t, = b.plot([], [], color=C[RECON], label="reality")
    line_d, = b.plot([], [], color=C[NORECON], linestyle=(0, (4, 2)), label="the dream")
    b.legend(loc="lower left")

    def draw(t: int):
        rod_t.set_data([0, true_xy[t, 0]], [0, true_xy[t, 1]])
        rod_d.set_data([0, dream_xy[t, 0]], [0, dream_xy[t, 1]])
        lo = max(0, t - 14)
        trail_t.set_data(true_xy[lo:t + 1, 0], true_xy[lo:t + 1, 1])
        trail_d.set_data(dream_xy[lo:t + 1, 0], dream_xy[lo:t + 1, 1])
        if t < CONTEXT:
            phase.set_text(f"filtering on observations, step {t + 1} of {CONTEXT}")
        else:
            phase.set_text(f"open loop step {t + 1 - CONTEXT}, "
                           f"angle gap {gap[t]:.1f} deg")
        line_t.set_data(steps[:t + 1], rew[:t + 1])
        line_d.set_data(steps[:t + 1], pred[:t + 1])
        return rod_t, rod_d, trail_t, trail_d, line_t, line_d, phase

    fig.tight_layout()
    frames = list(range(len(rew))) + [len(rew) - 1] * 10
    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 / fps, blit=False)
    # savefig.bbox is "tight" for the static figures, which would give the writer
    # a differently sized frame every time the text changes width.
    with matplotlib.rc_context({"savefig.bbox": None}):
        anim.save(out, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    made = [fig_open_loop(RESULTS / "open-loop.png"),
            fig_learning(RESULTS / "learning-curves.png"),
            fig_model_fit(RESULTS / "model-fit.png")]
    print("retraining the seed 0 world model for the animation, about a minute")
    try:
        made.append(anim_dream_vs_real(RESULTS / "dream-vs-real.gif"))
    except RuntimeError as e:
        print(f"!! animation skipped: {e}")
    for p in made:
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
