# world-model-from-scratch

A recurrent state-space model that learns environment dynamics from pixels, an actor-critic trained entirely inside it, and an honest study of the question the field keeps circling: **what does a world model actually need to model?**

> **Status: scaffold. Nothing here is built or measured yet.**
> This repo currently holds the project specification, the shared agent conventions,
> and an empty logbook. Every number in the tables below is a `TODO` because no
> experiment has been run. The `prompts/` task specs referenced in the wave table
> are not written yet either.
>
> Nothing in this repo is estimated or taken from a paper. When a table has a number
> in it, that number came from a run in `results/`.

---

## Why

Model-free RL learns a policy by acting in the world. A world model learns the world, then acts in imagination — thousands of simulated rollouts for every real environment step. The sample-efficiency gain is real and large.

The interesting question is representational. Dreamer reconstructs pixels and pays capacity for grass texture. MuZero reconstructs nothing and models only what's needed to predict value, reward, and policy. Contrastive methods sit between. **Task 05 runs that comparison directly**, which is the part of this repo that isn't just a reimplementation.

## Hardware

- **GPU:** `TODO — python -m scripts.env`
- DMC and Atari at 64×64 are modest — 12GB is fine. The video world model in task 06 is not; scale it down or skip it.
- Wall-clock is dominated by environment stepping, not the GPU. Vectorized envs matter more than a bigger card.

## Results

Sample efficiency — return at 100k environment steps, 3 seeds:

| Env | SAC/DQN (model-free) | RSSM + imagination | MuZero-style (no recon) | Contrastive |
|---|---:|---:|---:|---:|
| DMC walker-walk | TODO | TODO | TODO | TODO |
| DMC cheetah-run | TODO | TODO | TODO | TODO |
| Atari Pong | TODO | TODO | TODO | TODO |

Open-loop prediction horizon (steps before predicted and true observation diverge past threshold):

| Model | Horizon | Reward MAE @ 15 steps |
|---|---:|---:|

## Waves

```
00 bootstrap + envs + replay             (serial)
   ├─ 01 theory: POMDP, RSSM, KL balance ┐
   └─ 02 model-free baselines            ┘ parallel
        └─ 03 RSSM: learn the dynamics   (serial)
             ├─ 04 actor-critic in imagination ┐
             └─ 05 representation ablation     ┘ parallel
                  └─ 06 video world model + writeup
```

| Task | OWNS | READS |
|---|---|---|
| 00 | `scripts/`, `Makefile`, `wm/envs/`, `wm/buffer.py` | — |
| 01 | `notes/00-rssm.md`, `wm/ref/` | `scripts/` |
| 02 | `wm/modelfree/`, `train/train_mf.py` | `wm/envs/`, `wm/buffer.py` |
| 03 | `wm/rssm/`, `train/train_wm.py` | `wm/ref/`, `wm/buffer.py` |
| 04 | `wm/agent/`, `train/train_dreamer.py` | `wm/rssm/` |
| 05 | `experiments/representation/` | `wm/rssm/`, `wm/agent/` |
| 06 | `wm/video/`, `bench/`, `notes/paper.md`, `README.md` | everything |

See [`CONVENTIONS.md`](CONVENTIONS.md). Repo 06's RL machinery and repos 03/04's generative machinery both plug in here.

## Author

Aghasalim Mustafazada — third-year AI student at Howest, Belgium.

<p align="center">
  <a href="https://github.com/aghasalim">
    <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="github"></a>
  <a href="https://www.kaggle.com/aghasalimmustafazada">
    <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="kaggle"></a>
  <a href="https://linkedin.com/in/mustafazada">
    <img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="linkedin"></a>
  <a href="https://orcid.org/0009-0001-8746-4582">
    <img src="https://img.shields.io/badge/ORCID-A6CE39?style=for-the-badge&logo=orcid&logoColor=white" alt="orcid"></a>
</p>

## License

MIT — see [LICENSE](LICENSE).
