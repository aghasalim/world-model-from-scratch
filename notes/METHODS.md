# Methods and detail

Long form detail moved out of the README.


## What did not work


![learning curves](../results/learning-curves.png)

| method | return | range over seeds | env steps |
|---|---:|---|---:|
| recon (Dreamer style) | −38.13 | −40.9 to −36.8 | 57,600 |
| no-recon (MuZero style) | −39.17 | −40.4 to −35.9 | 57,600 |
| contrastive | −40.92 | −42.7 to −40.6 | 57,600 |
| model-free (recurrent PG) | −40.66 | −43.2 to −38.1 | 57,600 |
| model-free, run out to 384,000 steps | −35.08 | −40.0 to −34.9 | 384,000 |

A random policy scores about −38. Nothing here clears it by a margin worth
claiming, and the sample efficiency result the repo was supposed to demonstrate
does not reproduce.

I tuned this rather than giving up on the first failure. The imagination horizon
mattered most: at 15 steps, which is 0.75 seconds of simulated time, the policy
cannot see the payoff of swinging back to build momentum, and raising it to 40
took the best run from −40 to −30. I also added the slow moving target critic
that DreamerV2 uses and I had skipped, since the critic was regressing toward
targets built from its own output. Neither fix made it stable. The characteristic
failure is an oscillation: a run reaches −30.6 and then falls back to −39.4.

What I think is happening, without having proven it: the actor is trained by
backpropagating through imagined rollouts, so it optimises against the reward
head, and the reward head is only accurate near states the model has seen. As
the policy improves it moves into exactly the region where its own critic is
least reliable. That is the same failure this repo's sibling
[rlhf-ppo-from-scratch](https://github.com/aghasalim/rlhf-ppo-from-scratch)
measures deliberately, arriving from a completely different direction.

Establishing that properly would need a KL style constraint on how far the
policy may move per update, and a check of whether imagined returns match
realised ones. Both are the obvious next step and neither is done.


## What I got wrong


**I set the imagination horizon by copying a number rather than by thinking about
the task.** Fifteen steps is standard in Dreamer papers on control suites with
different timesteps. Here it was 0.75 seconds against a swing up that needs
several, so the agent was being asked to plan over a window in which the correct
action looks actively bad. That was three wasted debugging passes before I
computed what 15 steps meant in seconds.

**I omitted the target critic because the paper describes it as a stabilisation
detail.** It is, and adding it did not fix this particular failure, but I had no
basis for leaving it out other than that it looked optional. The tests now cover
both the target update and the fact that imagination never touches the encoder.
