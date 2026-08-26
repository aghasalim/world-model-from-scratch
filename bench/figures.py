"""Figures from committed CSVs. Nothing is re-run."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
C = {"recon (Dreamer style)": "#1a9850", "no-recon (MuZero style)": "#2166ac",
     "contrastive": "#b2182b", "model-free (recurrent PG)": "#999999"}


def fig_open_loop(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "open-loop.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13.5, 5.3))
    for mode in C:
        s = t[t["mode"] == mode]
        if s.empty:
            continue
        g = s.groupby("step")["reward_mae"]
        a.plot(g.median().index, g.median().values, color=C[mode], label=mode, linewidth=2)
        a.fill_between(g.median().index, g.min().values, g.max().values,
                       color=C[mode], alpha=0.15, linewidth=0)
    a.set_xlabel("open loop step (no observations after this point)")
    a.set_ylabel("reward prediction MAE")
    a.set_title("Rolling the prior forward with no observations\nmedian and range over 3 seeds")
    a.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=9)

    early = t[t["step"].isin([1, 2, 3, 5, 8])]
    for mode in C:
        s = early[early["mode"] == mode]
        if s.empty:
            continue
        g = s.groupby("step")["reward_mae"]
        b.plot(g.median().index, g.median().values, marker="o", color=C[mode],
               label=mode, linewidth=2)
        b.fill_between(g.median().index, g.min().values, g.max().values,
                       color=C[mode], alpha=0.15, linewidth=0)
    b.set_xlabel("open loop step")
    b.set_ylabel("reward prediction MAE")
    b.set_title("Short horizon, where the bands separate\n"
                "reconstruction's worst seed beats no-recon's best at k=1 and k=5")
    b.grid(alpha=0.3)
    b.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_learning(out: Path) -> Path:
    t = pd.read_csv(RESULTS / "learning-curves.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for mode in C:
        s = t[t["mode"] == mode]
        if s.empty:
            continue
        g = s.groupby("env_steps")["return"]
        ax.plot(g.median().index, g.median().values, color=C[mode], label=mode, linewidth=1.9)
        ax.fill_between(g.median().index, g.min().values, g.max().values,
                        color=C[mode], alpha=0.13, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode return")
    ax.set_title("Nothing here reliably solves the task.\n"
                 "All four sit in the same band and none of them settle.")
    ax.grid(alpha=0.3, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_model_fit(out: Path) -> Path:
    """The part that did work: the model itself gets better."""
    t = pd.read_csv(RESULTS / "learning-curves.csv")
    t = t[t["reward"].notna()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.9))
    for mode in ("recon (Dreamer style)", "no-recon (MuZero style)", "contrastive"):
        s = t[t["mode"] == mode]
        if s.empty:
            continue
        for ax, col in ((a, "reward"), (b, "kl")):
            g = s.groupby("iter")[col]
            ax.plot(g.median().index, g.median().values, color=C[mode], label=mode, linewidth=1.9)
            ax.fill_between(g.median().index, g.min().values, g.max().values,
                            color=C[mode], alpha=0.13, linewidth=0)
    a.set_yscale("log"); a.set_ylabel("reward prediction loss"); a.set_title("The model learns the dynamics")
    b.set_ylabel("KL(posterior || prior)"); b.set_title("KL rises as the posterior gets informative")
    for ax in (a, b):
        ax.set_xlabel("training iteration"); ax.grid(alpha=0.3, which="both")
        ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    for p in (fig_open_loop(RESULTS / "open-loop.png"),
              fig_learning(RESULTS / "learning-curves.png"),
              fig_model_fit(RESULTS / "model-fit.png")):
        print(f"-> {p.relative_to(ROOT)} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
