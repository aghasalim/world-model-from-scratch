# What ran

The committed `results/` came from one invocation of `experiments/main.py`.
This is that invocation taken apart: which flags moved off their defaults,
what each arm cost, and where the numbers that were never on the command
line are set. Each value has a file and line beside it.

## The invocation

    python -m experiments.main --seeds 0 1 2 --iters 60 --imag-horizon 40 --actor-lr 1e-3 --eval-every 5

That is the command under Running it in `README.md`. `verify/gocheck` reads
it back out of the README and compares each flag with
`results/run-meta.json` (`verify/gocheck/main.go:278` to `:326`), so the
two cannot drift apart without CI going red.

Four flags override a default in `experiments/main.py`:

| flag | default | this run | default at |
|---|---:|---:|---|
| `--iters` | 40 | 60 | `main.py:169` |
| `--imag-horizon` | 15 | 40 | `main.py:176` |
| `--actor-lr` | 8e-5 | 1e-3 | `main.py:178` |
| `--eval-every` | 4 | 5 | `main.py:182` |

Only the horizon change is explained anywhere: `notes/LOGBOOK.md`,
2026-08-26, 15 steps is 0.75 s of pendulum time at dt 0.05. The other three
have no logbook entry. `--actor-lr` also sets the critic's learning rate
(`main.py:92`).

The rest ran at their defaults, which `run-meta.json` records too:

| flag | value | `main.py` |
|---|---:|---|
| `--envs` | 16 | :170 |
| `--horizon` | 60 | :171 |
| `--buffer` | 400 | :172 |
| `--batch` | 32 | :173 |
| `--wm-updates` | 25 | :174 |
| `--actor-updates` | 8 | :175 |
| `--wm-lr` | 3e-4 | :177 |
| `--mf-lr` | 3e-4 | :179 |
| `--mf-iters` | 400 | :180 |
| `--mf-eval-every` | 10 | :181 |
| `--explore` | 0.3 | :183 |
| `--kl-balance` | 0.8 | :184 |

## Where the budget comes from

Environment steps per iteration are `envs * horizon` (`main.py:123`,
`:159`): 16 * 60 = 960.

- world model arms: 60 iterations * 960 = 57,600, the `env_steps` column of
  `results/summary.csv` for all nine world model rows.
- model-free arm: 400 iterations * 960 = 384,000, the same column for its
  three rows.

The world model arms are evaluated every 5 iterations and at the last one
(`main.py:142`), 13 points per seed. The model-free arm is evaluated every
10 (`main.py:160`), 41 points per seed. 13 * 9 + 41 * 3 = 240, the row
count of `results/learning-curves.csv`.

## Not on the command line

| value | where |
|---|---|
| RSSM: deter 64, stoch 16, hidden 128, kl_scale 1.0, free nats 0.1, min std 0.1 | `wm/rssm.py:41` to `:49` |
| InfoNCE temperature 0.1 | `wm/rssm.py:135` |
| actor and critic: two hidden layers of 128, ELU | `wm/agent.py:26`, `:37` |
| lambda return: gamma 0.99, lambda 0.95 | `wm/agent.py:47`, `:60` |
| target critic, Polyak tau 0.02 | `wm/agent.py:61` |
| grad norm clip 100 for model, actor and critic | `main.py:131`, `wm/agent.py:79`, `:85` |
| Adam for all three optimisers | `main.py:90` to `:92` |
| model-free policy: GRU of 64, gamma 0.99, entropy 1e-3, clip 10 | `wm/modelfree.py:18`, `:52`, `:64` |
| pendulum: dt 0.05, max torque 2.0, max speed 8.0, g 10, m 1, l 1 | `wm/envs.py:21` to `:26` |
| evaluation: 64 envs, seeded `seed + 100` | `main.py:143` |
| open loop: 128 episodes, 20 context steps, seeded `seed + 500` | `main.py:199` |
| first collection round random, later rounds add `explore` noise | `main.py:103`, `:118` |

## Seeds

`torch.manual_seed(seed)` opens every arm (`main.py:82`, `:150`). The
environment is seeded from `seed` and the collection generator from
`seed + 7` (`main.py:95`, `:96`), so within a seed all three world model
modes see the same first random rollout. The model-free arm builds a new
environment per iteration seeded `seed * 31 + it` (`main.py:156`).

## Cost

`wall_s` in `results/summary.csv`, three seeds per arm:

| arm | `params` | `wall_s`, min to max | sum |
|---|---:|---|---:|
| recon (Dreamer style) | 95,235 | 57.4 s to 65.4 s | 186.0 s |
| no-recon (MuZero style) | 84,609 | 58.8 s to 67.9 s | 185.7 s |
| contrastive | 111,489 | 57.3 s to 64.0 s | 182.0 s |
| model-free (recurrent PG) | 0 | 6.2 s to 7.5 s | 20.1 s |

`params` counts the RSSM only (`main.py:198`) and is written as 0 for the
model-free arm (`main.py:208`). The four sums give 573.8 s; `run-meta.json`
has `wall_clock_s` 574.49 for the whole process. M4 laptop CPU
(`README.md`, Running it), `device` cpu, torch 2.13.0 (`run-meta.json`).
Dated 2026-08-26 by the logbook and by the first commit of `results/`.

## What it wrote

`results/learning-curves.csv` (240 rows), `results/open-loop.csv` (360
rows: 40 open loop steps, 3 modes, 3 seeds), `results/summary.csv` (12
rows) and `results/run-meta.json`, at `main.py:212` to `:222`. Nothing
else. There is no `torch.save` in `wm/` or `experiments/`, so the trained
RSSMs do not outlive the process. That is why the gif retrains the seed 0
model instead of loading it (`README.md`, Running it).
