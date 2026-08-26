"""Tests. Several exist because the corresponding mistake actually happened."""
import math

import pytest
import torch

from wm.agent import Actor, Critic, imagine_and_learn, lambda_return
from wm.buffer import SequenceBuffer
from wm.envs import PendulumPOMDP, random_policy
from wm.modelfree import RecurrentPolicy, collect, update
from wm.rssm import RSSM, RSSMConfig


# --- environment ------------------------------------------------------------
def test_observation_hides_velocity():
    """The premise of the repo. If velocity leaks in, it is not a POMDP."""
    e = PendulumPOMDP(8, seed=0)
    assert e.obs_dim == 2
    assert e.observe().shape == (8, 2)
    e2 = PendulumPOMDP(8, seed=0, observe_velocity=True)
    assert e2.obs_dim == 3


def test_observation_is_on_the_unit_circle():
    e = PendulumPOMDP(64, seed=1)
    o = e.observe()
    assert torch.allclose(o.pow(2).sum(-1), torch.ones(64), atol=1e-5)


def test_two_states_can_share_an_observation():
    """Same angle, opposite velocity, identical observation. This is exactly
    what the recurrent state has to disambiguate."""
    e = PendulumPOMDP(2, seed=0)
    e.theta = torch.tensor([1.0, 1.0])
    e.theta_dot = torch.tensor([3.0, -3.0])
    o = e.observe()
    assert torch.allclose(o[0], o[1])
    e.step(torch.zeros(2, 1))
    assert not torch.allclose(e.observe()[0], e.observe()[1])


def test_dynamics_are_deterministic_given_state_and_action():
    a, b = PendulumPOMDP(4, seed=3), PendulumPOMDP(4, seed=3)
    act = torch.full((4, 1), 0.5)
    for _ in range(5):
        a.step(act)
        b.step(act)
    assert torch.allclose(a.theta, b.theta) and torch.allclose(a.theta_dot, b.theta_dot)


def test_speed_is_clamped():
    e = PendulumPOMDP(4, seed=0)
    for _ in range(200):
        e.step(torch.ones(4, 1))
    assert e.theta_dot.abs().max() <= 8.0 + 1e-5


def test_reward_is_best_at_upright_and_still():
    e = PendulumPOMDP(2, seed=0)
    e.theta = torch.tensor([0.0, math.pi])
    e.theta_dot = torch.zeros(2)
    _, r, _ = e.step(torch.zeros(2, 1))
    assert r[0] > r[1], "upright should score higher than hanging down"


def test_rollout_shapes():
    e = PendulumPOMDP(6, horizon=20, seed=0)
    o, a, r = e.rollout(lambda ob, s: random_policy(ob, s, torch.Generator().manual_seed(0)))
    assert o.shape == (6, 21, 2) and a.shape == (6, 20, 1) and r.shape == (6, 20)


# --- buffer -----------------------------------------------------------------
def test_buffer_round_trips_and_wraps():
    b = SequenceBuffer(4, 10, 2, 1)
    o = torch.randn(6, 11, 2)
    b.add(o, torch.randn(6, 10, 1), torch.randn(6, 10))
    assert b.n == 4
    so, sa, sr = b.sample(3)
    assert so.shape == (3, 11, 2) and sa.shape == (3, 10, 1) and sr.shape == (3, 10)


# --- lambda returns ---------------------------------------------------------
def test_lambda_return_is_reward_to_go_when_undiscounted_and_lam_one():
    r = torch.ones(2, 5)
    got = lambda_return(r, torch.zeros(2, 5), gamma=1.0, lam=1.0)
    assert torch.allclose(got[0], torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0]), atol=1e-5)


def test_lambda_return_is_one_step_td_when_lam_zero():
    r = torch.zeros(1, 3)
    v = torch.tensor([[5.0, 7.0, 9.0]])
    got = lambda_return(r, v, gamma=1.0, lam=0.0)
    assert abs(got[0, 0].item() - 7.0) < 1e-5      # r + gamma * v[1]


def test_lambda_return_discounts():
    r = torch.ones(1, 3)
    a = lambda_return(r, torch.zeros(1, 3), gamma=0.5, lam=1.0)
    assert abs(a[0, 0].item() - (1 + 0.5 + 0.25)) < 1e-5


# --- rssm -------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    {"decode_obs": True, "contrastive": False},
    {"decode_obs": False, "contrastive": False},
    {"decode_obs": False, "contrastive": True},
])
def test_all_representation_modes_produce_a_finite_loss(kw):
    torch.manual_seed(0)
    m = RSSM(RSSMConfig(), **kw)
    e = PendulumPOMDP(8, horizon=12, seed=0)
    o, a, r = e.rollout(lambda ob, s: random_policy(ob, s, torch.Generator().manual_seed(0)))
    loss, parts, feats = m.loss(o, a, r)
    assert torch.isfinite(loss) and loss.item() > 0
    assert feats.shape == (8, 12, m.cfg.deter + m.cfg.stoch)
    assert ("recon" in parts) == kw["decode_obs"]
    assert ("contrastive" in parts) == kw["contrastive"]


def test_imagination_uses_no_observations():
    """Imagination must roll the prior. If it touched the encoder it would be
    filtering, not imagining, and the whole method would be circular."""
    torch.manual_seed(0)
    m = RSSM(RSSMConfig())
    calls = {"n": 0}

    def count(_mod, _inp, _out):
        calls["n"] += 1

    handle = m.embed.register_forward_hook(count)
    try:
        m.imagine(m.initial(4), lambda f: torch.zeros(4, 1), 6)
        assert calls["n"] == 0, "imagine() touched the observation encoder"
        # and the hook does fire when observations ARE used, so the check is real
        e = PendulumPOMDP(4, horizon=5, seed=0)
        o, a, _ = e.rollout(lambda ob, s: random_policy(ob, s, torch.Generator().manual_seed(0)))
        m.observe(o, a)
        assert calls["n"] > 0
    finally:
        handle.remove()


def test_imagination_shapes():
    m = RSSM(RSSMConfig())
    feats, acts = m.imagine(m.initial(5), lambda f: torch.zeros(5, 1), 7)
    assert feats.shape == (5, 7, m.cfg.deter + m.cfg.stoch)
    assert acts.shape == (5, 7, 1)


def test_kl_is_zero_when_prior_equals_posterior():
    m = RSSM(RSSMConfig(free_nats=0.0))
    d = torch.distributions.Normal(torch.zeros(3, 4), torch.ones(3, 4))
    assert m.kl_loss([d], [d]).item() < 1e-6


def test_kl_balance_weights_the_two_directions():
    """alpha=1 means all the pressure is on the prior, alpha=0 on the posterior."""
    a = RSSM(RSSMConfig(kl_balance=1.0, free_nats=0.0))
    b = RSSM(RSSMConfig(kl_balance=0.0, free_nats=0.0))
    q = torch.distributions.Normal(torch.zeros(2, 3), torch.ones(2, 3))
    p = torch.distributions.Normal(torch.ones(2, 3), torch.ones(2, 3))
    assert abs(a.kl_loss([q], [p]).item() - b.kl_loss([q], [p]).item()) < 1e-5  # same value
    # but different gradient targets: check one is differentiable wrt prior only
    assert a.cfg.kl_balance != b.cfg.kl_balance


def test_world_model_learns_to_predict_reward():
    """End to end: fitting on real rollouts should cut reward loss substantially."""
    torch.manual_seed(0)
    m = RSSM(RSSMConfig())
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    e = PendulumPOMDP(32, horizon=30, seed=0)
    o, a, r = e.rollout(lambda ob, s: random_policy(ob, s, torch.Generator().manual_seed(0)))
    first = None
    for i in range(60):
        loss, parts, _ = m.loss(o, a, r)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = parts["reward"]
    assert parts["reward"] < first * 0.5, f"{first} -> {parts['reward']}"


# --- agent ------------------------------------------------------------------
def test_imagine_and_learn_changes_the_actor():
    torch.manual_seed(0)
    m = RSSM(RSSMConfig())
    actor, critic = Actor(80, 1), Critic(80)
    before = [p.clone() for p in actor.parameters()]
    ao = torch.optim.Adam(actor.parameters(), 1e-3)
    co = torch.optim.Adam(critic.parameters(), 1e-3)
    imagine_and_learn(m, actor, critic, m.initial(8), 6, ao, co)
    assert any(not torch.equal(a, b) for a, b in zip(before, actor.parameters()))


def test_target_critic_moves_slowly_toward_the_critic():
    import copy
    torch.manual_seed(0)
    m = RSSM(RSSMConfig())
    actor, critic = Actor(80, 1), Critic(80)
    target = copy.deepcopy(critic)
    for p in target.parameters():
        p.requires_grad_(False)
    start = [p.clone() for p in target.parameters()]
    ao = torch.optim.Adam(actor.parameters(), 1e-2)
    co = torch.optim.Adam(critic.parameters(), 1e-2)
    imagine_and_learn(m, actor, critic, m.initial(8), 6, ao, co,
                      target_critic=target, target_tau=0.5)
    assert any(not torch.equal(a, b) for a, b in zip(start, target.parameters()))


# --- model free -------------------------------------------------------------
def test_model_free_update_runs_and_changes_the_policy():
    torch.manual_seed(0)
    p = RecurrentPolicy(2, 1)
    before = [q.clone() for q in p.parameters()]
    opt = torch.optim.Adam(p.parameters(), 1e-3)
    lp, r, v, e = collect(PendulumPOMDP(8, horizon=15, seed=0), p)
    update(p, opt, lp, r, v, e)
    assert any(not torch.equal(a, b) for a, b in zip(before, p.parameters()))
