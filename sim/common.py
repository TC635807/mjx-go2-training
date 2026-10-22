#!/usr/bin/env python
"""Shared helpers used by the evaluation / probing scripts.

`load_policy` loads a brax PPO policy either from a pickled params file
(`policies/*.pkl`) or from an orbax checkpoint directory
(`logs/*/checkpoints/<step>`). Network sizes and observation dimensions are
inferred from the parameter shapes, so it works for both the walk and the
get-up policies without extra configuration.
"""
import json
import os

import jax


def infer_layers_and_dims(params):
  roots = [params]
  if isinstance(params, (tuple, list)) and len(params) > 1:
    roots = [params[1]]
  found = {}

  def walk(node):
    if isinstance(node, (tuple, list)):
      for v in node:
        walk(v)
      return
    if not hasattr(node, "items"):
      return
    for k, v in node.items():
      if (k.startswith("hidden_") and hasattr(v, "get") and "kernel" in v
          and hasattr(v["kernel"], "shape")):
        found[int(k.split("_")[1])] = int(v["kernel"].shape[1])
      walk(v)

  for r in roots:
    walk(r)
  layers = tuple(found[i] for i in sorted(found)) if found else (128, 128)

  def first_dim(node):
    best = {}

    def w(n):
      if isinstance(n, (tuple, list)):
        for v in n:
          w(v)
        return
      if not hasattr(n, "items"):
        return
      for k, v in n.items():
        if (k == "hidden_0" and hasattr(v, "get") and "kernel" in v
            and hasattr(v["kernel"], "shape")):
          best.setdefault("dim", int(v["kernel"].shape[0]))
        w(v)

    w(node)
    return best.get("dim")

  if isinstance(params, (tuple, list)) and len(params) >= 3:
    ns, nv = first_dim(params[1]) or 48, first_dim(params[2]) or 119
  else:
    ns, nv = (first_dim(params) or 48), 119
  return layers, ns, nv


def load_policy(path):
  from brax.training.agents.ppo import networks as ppo_networks
  if os.path.isdir(path):
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    cfg = json.load(open(os.path.join(path, "ppo_network_config.json")))
    kw = cfg["network_factory_kwargs"]
    params = ppo_ckpt.load(path)
    layers, ns, nv = infer_layers_and_dims(params)
    net = ppo_networks.make_ppo_networks(
        observation_size={"state": (ns,), "privileged_state": (nv,)},
        action_size=cfg["action_size"],
        policy_hidden_layer_sizes=tuple(kw["policy_hidden_layer_sizes"]),
        value_hidden_layer_sizes=tuple(kw["value_hidden_layer_sizes"]),
        policy_obs_key=kw["policy_obs_key"], value_obs_key=kw["value_obs_key"],
        distribution_type=kw.get("distribution_type", "tanh_normal"),
        activation=jax.nn.tanh)
  else:
    import pickle
    with open(path, "rb") as f:
      params = pickle.load(f)
    layers, ns, nv = infer_layers_and_dims(params)
    net = ppo_networks.make_ppo_networks(
        observation_size={"state": (ns,), "privileged_state": (nv,)},
        action_size=12, policy_hidden_layer_sizes=layers,
        value_hidden_layer_sizes=layers, policy_obs_key="state",
        value_obs_key="privileged_state", distribution_type="normal",
        activation=jax.nn.tanh)
  return ppo_networks.make_inference_fn(net)(params, deterministic=True), ns
