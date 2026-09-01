# Logbook

## 2026-09-01, the random baseline I had been reading everything against was wrong
**Tried:** reimplementing the environment in Rust so a second implementation would have to agree with `wm/envs.py` before I trusted either. See `verify/pendulum`.
**Measured:** a uniform random policy over 200,000 episodes returns -36.87, standard error 0.02. PyTorch over the same number of episodes gives -36.85. I had been quoting -38 everywhere, from the earlier entry below onwards.
**Concluded:** the figure line, the README and `notes/METHODS.md` all moved to -36.9. It makes the negative result stronger rather than weaker: at 57,600 environment steps every world model here is below random, not level with it. The number had never been measured on its own, only eyeballed off a learning curve, and nothing in the repository recomputed it. That is exactly the gap `verify/` exists to close.

## 2026-08-26, the imagination horizon was set by copying a number
**Tried:** first full run. The world model fit well (reward loss 0.0713 down to 0.0047, about fifteen times) but the policy did not move: return stayed between -38 and -40 for sixty iterations, which is where a random policy sits.
**Measured:** raising the imagination horizon from 15 to 40 took the best run from -40.1 to -30.1. Raising it further to 50 was worse again.
**Concluded:** 15 is the number Dreamer papers use on control suites whose timestep is not this one. At dt=0.05 it is 0.75 seconds, and a pendulum swing up needs several: the agent was being asked to plan over a window in which swinging away from the target looks purely bad, because the payoff falls outside the horizon. Three debugging passes before I worked out what 15 steps meant in seconds. Worth converting horizons into task time before trusting a hyperparameter transplanted from another paper.

## 2026-08-26, the target critic was not the fix either
**Tried:** the policy oscillated, reaching -30.6 then falling back to -39.4. Suspected the critic bootstrapping off its own current output, so added the slow moving target network DreamerV2 uses and I had skipped.
**Measured:** with the target critic, runs still oscillate: -40.3, -39.3, -39.1, -33.6, -41.0.
**Concluded:** correct thing to add and it did not solve this. My current guess, unproven: the actor is trained by backpropagating through imagined rollouts, so it optimises against the reward head, and the reward head is accurate only near states the model has actually seen. As the policy improves it walks into the region where its own model is least reliable. That is the same failure the sibling RLHF repo measures on purpose from a different direction. Testing it would need a trust region on the policy update and a comparison of imagined against realised returns; neither is done, and the README says so rather than implying the algorithm is fine.

## 2026-08-26, reconstruction helps for five steps and then stops mattering
**Tried:** the representation ablation. Three RSSMs trained identically, differing only in whether they reconstruct observations, predict reward alone, or use an InfoNCE term. Measured by filtering 20 steps then rolling the prior open loop with no observations. 3 seeds, 574 s total.
**Measured:** reward MAE at k=1 is 0.0403 (recon), 0.0589 (no-recon), 0.1058 (contrastive). The seed values do not overlap at k=1: recon is 0.0360/0.0403/0.0439 and no-recon is 0.0535/0.0589/0.0889, so the worst recon seed beats the best no-recon seed. Same at k=5, 0.0578 against 0.0674. By k=10 they overlap (recon 0.0699 to 0.1203, no-recon 0.0853 to 0.1883) and by k=40 the medians are 0.2910 and 0.2884.
**Concluded:** a real but narrow result. The decoder buys accuracy while the state is still anchored to recent observations and buys nothing once the rollout has drifted, which is a smaller claim than "reconstruction is worth it" and the one the data supports. Contrastive being worst everywhere surprised me: its InfoNCE term only has to identify which observation in the batch a latent belongs to, and on a two dimensional observation on the unit circle that is easy without encoding much. A richer observation space would likely change this and I would not generalise from one toy.

## 2026-08-26, no method solved the task, including the baseline
**Tried:** full comparison at matched environment steps.
**Measured:** at 57,600 steps: recon -38.13, no-recon -39.17, contrastive -40.92, model free -40.66. Random is about -38. Model free run out to 384,000 steps reaches -35.08.
**Concluded:** the sample efficiency demonstration this repo was meant to produce does not reproduce, and reporting the world model as ahead on the basis of -38.13 against -40.66 would be reading noise, since the seed ranges overlap completely. The parts that did work, the model fit and the open loop ablation, are reported as the result and the policy half is reported as a failure with what I tried.
