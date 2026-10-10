# world-model-from-scratch

[![ci](https://github.com/aghasalim/world-model-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/aghasalim/world-model-from-scratch/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![results](https://img.shields.io/badge/results-reproducible-1a9850.svg)](results/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23003689.svg)](https://doi.org/10.5281/zenodo.23003689)

I built an RSSM from the Dreamer papers and trained an actor critic entirely in
imagination. I also ran an ablation to see if a world model needs to reconstruct
observations at all.

Half of this worked. The world model learns, and I got a real answer on the
representation question. The agent trained inside it didn't work at the budget I
planned for. At 57,600 environment steps none of the world models beats a random
policy. With the 384,000 steps the model-free baseline got, all three beat random
and the baseline too. Still, none of them solves the task, and one run collapses.
I kept the half that failed because of [`METHODOLOGY.md`](METHODOLOGY.md) rule 12,
and that's a big part of why this README looks the way it does.

I didn't use a GPU. The main sweep takes ten minutes on my laptop CPU, and the
384,000 step runs take another 23 minutes. Every number published
here is recomputed from the committed results by separate implementations in
`verify/`, and CI fails if any of them disagree.

## The task

It's a pendulum swing up where the agent only sees `cos(theta)` and `sin(theta)`.
The angular velocity is hidden.

That's why I wrote the environment by hand. If the agent could see velocity, it
would be a plain MDP that a feedforward policy can solve, and nothing would test
whether the model keeps track of state. Hiding velocity makes it a POMDP. Two
states with the same angle and opposite velocity give the same observation, and
the only way to tell them apart is to remember where the pendulum was. That's
what the recurrent state of an RSSM is for. There's also a test that checks the
ambiguity is really there.

## The representation ablation

Dreamer reconstructs observations and spends capacity on that. MuZero style
models don't reconstruct anything and only predict what control needs.
Contrastive methods are somewhere in between. I put all three behind one flag,
trained them the same way, and measured how far each can roll forward without
any observations.

![open loop prediction error](results/open-loop.png)

Reward prediction MAE after filtering 20 steps and then predicting open loop,
median over 3 seeds:

| open loop step | recon (Dreamer) | no-recon (MuZero) | contrastive |
|---:|---:|---:|---:|
| 1 | **0.0403** | 0.0589 | 0.1058 |
| 5 | **0.0577** | 0.0716 | 0.1034 |
| 10 | 0.0966 | 0.0995 | 0.1849 |
| 15 | 0.1635 | 0.1843 | 0.2799 |
| 20 | 0.2230 | 0.3029 | 0.2954 |
| 40 | 0.2910 | 0.2884 | 0.3648 |

Reconstruction helps, and at short horizons it's clear. At one step the three
reconstruction seeds are 0.0360, 0.0403 and 0.0439. The three no-recon seeds are
0.0535, 0.0589 and 0.0889. Even the worst reconstruction seed beats the best
no-recon seed, so the bands don't overlap. It's the same at k=5, with 0.0578
worst against 0.0674 best.

By ten steps the advantage is gone. At k=10 recon spans 0.0699 to 0.1203 and
no-recon spans 0.0853 to 0.1883, so they overlap. By k=40 the medians are 0.2910
and 0.2884, which I can't tell apart. Recon has degraded to 7.2 times its one
step error by then and no-recon to 4.9 times. No-recon started worse at one step, so
it had less room to get worse.

So reconstruction makes the model more accurate while it's still close to recent
observations, and stops helping once the trajectory has drifted. That's a
narrower claim than "Dreamer's decoder is worth it", but it's the one my
experiment supports.

![imagined rollout against the true rollout](results/dream-vs-real.gif)

Both pendulums start from the same twenty observed steps and get the same action
sequence. After step 20 the model gets nothing. It keeps the shape of the swing
but loses the timing. This is the reconstruction model at seed 0, on the episode
whose open loop error is the median of the 128 the numbers above are averaged
over.

Contrastive was the worst at every horizon, and that surprised me. Its InfoNCE
term only asks the latent to pick out which observation in the batch it belongs
to. With a two dimensional observation on the unit circle that's an easy task,
so the latent doesn't have to encode much. I'd guess it would do better with a
richer observation space.

## The model itself learns

![model fitting](results/model-fit.png)

From the median first iteration loss to the median last one, reward prediction
loss falls by 23 times for recon, 23 times for no-recon and 35 times for
contrastive. The KL between posterior and prior goes up as the posterior starts
carrying information, which is what I expected.

## What did not work

| method | return | range over seeds | env steps |
|---|---:|---|---:|
| recon (Dreamer style) | −38.13 | −40.9 to −36.8 | 57,600 |
| no-recon (MuZero style) | −39.17 | −40.4 to −35.9 | 57,600 |
| contrastive | −40.92 | −42.7 to −40.6 | 57,600 |
| model-free (recurrent PG) | −40.66 | −43.2 to −38.1 | 48,960 |
| model-free, run out to 384,000 steps | −35.08 | −40.0 to −34.9 | 384,000 |

A random policy scores about −36.9, measured over 200,000 episodes by
`verify/pendulum`. That beats every world model in the table, so the sample
efficiency result I set out to show doesn't reproduce. Put plainly, at 57,600 environment steps
none of the three world models has learned anything useful. The best one, at
−38.13, is below random. The model-free baseline at −40.66 is further below. The
gap between those two is much smaller than the spread across three seeds.

The model-free arm is evaluated every ten iterations and never lands exactly on
57,600 steps. Its row above is the nearest evaluation below that, at 48,960. The
next one up, at 58,560 steps, has a median of −35.89, which beats every world
model final here. Picking a different evaluation point moves the comparison more
than picking a different method does, so it points to the same conclusion.

When I ran that baseline out to 384,000 steps it reached −35.08. That's better
than every final in the table above, and it's the only entry that beats a random
policy at all. Tuning changed things without fixing them. Raising the imagination
horizon from 15 to 40 took the best run from −40.1 to −30.1, and then the runs oscillate: one reaches −30.6 and falls back to −39.4.

### Given the same 384,000 steps

The comparison above isn't fair to the world models. I ran the model-free
baseline out to 384,000 environment steps but stopped them at 57,600. So I reran
each one for 400 iterations, 384,000 steps, and left every other setting the same ([`results/long-summary.csv`](results/long-summary.csv),
[`results/long-curves.csv`](results/long-curves.csv)).

| method, at 384,000 steps | seed 0 | seed 1 | seed 2 | median |
|---|---:|---:|---:|---:|
| recon (Dreamer style) | −28.54 | −30.81 | −31.75 | −30.81 |
| no-recon (MuZero style) | −31.15 | −20.51 | −26.06 | −26.06 |
| contrastive | −32.26 | −32.33 | −40.91 | −32.33 |
| model-free (recurrent PG) | −40.03 | −34.94 | −35.08 | −35.08 |

With the same experience, every world model beats the random policy's −36.9 and
the model-free baseline. The one exception is a contrastive run that goes flat at
−40.91 by about 25,000 steps and never moves again. No-recon is best on two of three seeds and has the best
single run, −20.51. So the sample efficiency result is real. It just needs more
steps than the budget I originally planned for.

I still wouldn't call this a win, for two reasons. None of these returns comes
close to holding the pendulum up, and the curves still wander. The best return any
world model hit during training was −16.82, on a seed that finished at −26.06.
Also, counting per environment step isn't the same as counting compute. A 384,000
step world model run took about 420 s here, and the model-free one took about 7 s.

![learning curves](results/learning-curves.png)

Longer version in [notes/METHODS.md](notes/METHODS.md#what-did-not-work).

## What the checks caught

The checks found two mistakes. I had quoted the random policy baseline as about
−38 everywhere, including the dashed line every learning curve is compared to. I'd
never actually measured it by itself. Two separate reimplementations of the
environment, 200,000 episodes each, give −36.87 and −36.85, so it's now −36.9.
That made the negative result stronger, since at 57,600
environment steps every world model here is below random.

The second one is smaller. I had labelled the model-free row of the return table
57,600 environment steps. But the model-free arm is only evaluated every ten
iterations, and its nearest evaluation is 48,960. The row says 48,960 now. I also
mention what the next evaluation up gives, because only showing the one below
makes the world models look better.

## What I got wrong

I picked the imagination horizon by copying a number instead of thinking about
the task. Fifteen steps is standard in the Dreamer papers, but their control
suites use different timesteps. At dt=0.05 that is 0.75 seconds, and a swing up
takes several. So I was asking the agent to plan over a window where the right
action actually looks bad. It took me three rounds of debugging before I
converted 15 steps into seconds.

I also left out the target critic because DreamerV2 describes it as a
stabilisation detail. Putting it back didn't fix the oscillation either.

Longer version in [notes/METHODS.md](notes/METHODS.md#what-i-got-wrong).

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python -m pytest tests/ -q
```

```bash
python -m experiments.main --seeds 0 1 2 --iters 60 --imag-horizon 40 --actor-lr 1e-3 --eval-every 5
```

```bash
python -m bench.figures
```

The sweep takes about ten minutes on an M4 CPU and writes `results/*.csv`. The
three static figures only read those files and never re-run an experiment. The
animation is different, because a summary file can't hold a drifting rollout. It
retrains the seed 0 world model, which takes about a minute, and won't write
itself unless its open loop error still matches the committed `open-loop.csv`
exactly.

[notes/FLAGS.md](notes/FLAGS.md) goes through that command flag by flag and
says what each arm cost.

## What is where

```
wm/envs.py       the POMDP pendulum, written directly, no simulator dependency
wm/rssm.py       deterministic GRU path, stochastic latent, KL balancing
wm/agent.py      actor critic in imagination, lambda returns, target critic
wm/modelfree.py  recurrent policy gradient baseline, same architecture
wm/buffer.py     sequence replay
experiments/     the sweep and the open loop measurement
verify/          the same numbers recomputed independently
tests/           22 tests
```

## Reading

- **Hafner, Lillicrap, Fischer, Villegas, Ha, Lee, Davidson. Learning Latent Dynamics for Planning from Pixels. ICML 2019.** [arXiv:1811.04551](https://arxiv.org/abs/1811.04551) PlaNet, and the RSSM's split of the latent into deterministic and stochastic parts.
- **Hafner, Lillicrap, Ba, Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. ICLR 2020.** [arXiv:1912.01603](https://arxiv.org/abs/1912.01603) Dreamer: backpropagating the actor through imagined trajectories.
- **Hafner, Lillicrap, Norouzi, Ba. Mastering Atari with Discrete World Models. ICLR 2021.** [arXiv:2010.02193](https://arxiv.org/abs/2010.02193) DreamerV2. KL balancing, lambda returns and the target critic all come from here.
- **Schrittwieser, Antonoglou, Hubert et al. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. Nature 2020.** [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) MuZero: the no-reconstruction position this ablation tests.
- **Laskin, Srinivas, Abbeel. CURL: Contrastive Unsupervised Representations for Reinforcement Learning. ICML 2020.** [arXiv:2004.04136](https://arxiv.org/abs/2004.04136) The contrastive arm.
- **Kaelbling, Littman, Cassandra. Planning and Acting in Partially Observable Stochastic Domains. AI 1998.** Why hiding velocity changes the problem, not just its difficulty.

## License

MIT, see [LICENSE](LICENSE).
