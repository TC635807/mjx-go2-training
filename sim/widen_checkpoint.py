#!/usr/bin/env python
"""把 1024-512 brax checkpoint 用 net2wider 扩成 1536-768, 生成可续训的新 checkpoint。

net2wider 保证函数近似不变 (逐层复制隐藏单元并 1/c 缩放):
  h_new[j] = pre_old[g(j)] / c(g(j)),  g: new->old 均匀映射, c: 每个 old 单元被复制次数
  W1_new[i,j] = W1_old[i,g(j)]/c(g(j)); b1_new[j] = b1_old[g(j)]/c(g(j))
  W2_new[j,m] = W2_old[g(j),g2(m)]/c2(g2(m)); b2_new[m] = b2_old[g2(m)]/c2(g2(m))
  Wout_new[m,k] = Wout_old[g2(m),k] (复制行, 不再缩放); bias 不变
用法: python sim/widen_checkpoint.py <old_ckpt_dir> <new_ckpt_root> <new_h0> <new_h1>
输出: <new_ckpt_root>/000000000000 (含 ppo_network_config.json), 供 --restore 使用
"""
import json
import os
import sys

import numpy as np

from brax.training import checkpoint
import jax


def _map_counts(n_new, n_old):
    g = (np.arange(n_new) * n_old) // n_new          # new -> old
    c = np.bincount(g, minlength=n_old).astype(np.float32)
    return g, np.maximum(c, 1.0)


def widen_hidden(kern, bias, g_prev, c_prev, g_cur, c_cur, first=False):
    """kern: (n_in_old, n_out_old); 返回 (n_in_new, n_out_new)。

    net2wider 的正确形式 (激活是 tanh, 缩放必须放在**激活之后**进入下一层的权重上):
    * 输出单元复制:      W_new[:, m] = W_old[:, g(m)],  b_new[m] = b_old[g(m)]
    * 输入单元复制:      W_new[j, :] = W_old[g_prev(j), :] / c_prev[g_prev(j)]
    这样 new 层每个神经元的 pre-activation / tanh 输出都与被复制的 old 单元一致,
    下一层求和时每 old 单元出现 c 次、权重除 c -> 函数不变。
    """
    k = np.asarray(kern, dtype=np.float32)
    b = np.asarray(bias, dtype=np.float32)
    k2 = k[:, g_cur]
    b2 = b[g_cur]
    if not first:
      k2 = k2[g_prev, :] / c_prev[g_prev][:, None]
    return k2, b2


def widen_mlp(params, new_sizes):
    p = params["params"]
    # 找 hidden 层 (按名字排序)
    hnames = sorted([k for k in p if k.startswith("hidden_")])
    old_sizes = [p[hnames[0]]["kernel"].shape[1]]
    for h in hnames[1:]:
      old_sizes.append(p[h]["kernel"].shape[1])
    assert len(old_sizes) == len(new_sizes), (old_sizes, new_sizes)
    gs, cs = [], []
    for n_new, n_old in zip(new_sizes, old_sizes):
      g, c = _map_counts(n_new, n_old)
      gs.append(g)
      cs.append(c)
    out = {}
    for i, h in enumerate(hnames):
      kern = p[h]["kernel"]
      bias = p[h]["bias"]
      k2, b2 = widen_hidden(
          kern, bias,
          gs[i - 1] if i > 0 else None,
          cs[i - 1] if i > 0 else None,
          gs[i], cs[i], first=(i == 0))
      out[h] = {"kernel": k2, "bias": b2}
    return out


def main():
    old = os.path.abspath(sys.argv[1])
    new_root = os.path.abspath(sys.argv[2])
    new_h = [int(v) for v in sys.argv[3:5]]
    params = checkpoint.load(old)
    # brax checkpoint.load_config 在 kernel init fn 为 None 时会 KeyError(None),
    # 这里手动读 json 并还原 callable 字段。
    from ml_collections import config_dict
    from brax.training import networks as _networks
    _raw = json.load(open(os.path.join(old, "ppo_network_config.json")))
    _nf = _raw["network_factory_kwargs"]
    if isinstance(_nf.get("activation"), str):
      _nf["activation"] = _networks.ACTIVATION[_nf["activation"]]
    for _k in ("mean_kernel_init_fn", "policy_network_kernel_init_fn",
               "value_network_kernel_init_fn"):
      _v = _nf.get(_k)
      if isinstance(_v, str):
        _nf[_k] = _networks.KERNEL_INITIALIZER[_v]
    cfg = config_dict.ConfigDict(_raw)
    pol, val = params[1], params[2]
    # ---- policy: MLP + 输出层 Dense_0 ----
    pol_p = pol["params"]
    hnames = sorted([k for k in pol_p if k.startswith("MLP_0")] )
    mlp = {k.split("/")[-1]: v for k, v in [
        (kk, vv) for kk, vv in []]} if False else None
    # 直接把整个 MLP_0 子 dict 展开成 widener 期望的 {'params': {hidden_*}}
    mlp_sub = pol_p["MLP_0"]
    old_out = pol_p["Dense_0"]["kernel"].shape[0]
    new_out = int(round(old_out * new_h[1] / new_h[0]))  # 保持比例, 这里应等于 768
    gs, cs = [], []
    old_sizes = [mlp_sub["hidden_0"]["kernel"].shape[1],
                 mlp_sub["hidden_1"]["kernel"].shape[1]]
    for n_new, n_old in zip(new_h, old_sizes):
      g, c = _map_counts(n_new, n_old)
      gs.append(g); cs.append(c)
    pol_new = {"params": {"MLP_0": {}}}
    for i, h in enumerate(["hidden_0", "hidden_1"]):
      k2, b2 = widen_hidden(mlp_sub[h]["kernel"], mlp_sub[h]["bias"],
                            gs[i - 1] if i > 0 else None,
                            cs[i - 1] if i > 0 else None,
                            gs[i], cs[i], first=(i == 0))
      pol_new["params"]["MLP_0"][h] = {"kernel": k2, "bias": b2}
    # 输出层/标准差
    dk = np.asarray(pol_p["Dense_0"]["kernel"])
    db = np.asarray(pol_p["Dense_0"]["bias"])
    if dk.shape[0] != new_h[1]:
      dk = dk[gs[1], :] / cs[1][gs[1]][:, None]
    pol_new["params"]["Dense_0"] = {"kernel": dk, "bias": db}
    pol_new["params"]["std_param"] = pol_p["std_param"]
    # ---- value ----
    val_p = val["params"]
    vh = ["hidden_0", "hidden_1"]
    vold = [val_p[vh[0]]["kernel"].shape[1], val_p[vh[1]]["kernel"].shape[1]]
    vgs, vcs = [], []
    for n_new, n_old in zip(new_h, vold):
      g, c = _map_counts(n_new, n_old)
      vgs.append(g); vcs.append(c)
    val_new = {"params": {}}
    for i, h in enumerate(vh):
      k2, b2 = widen_hidden(val_p[h]["kernel"], val_p[h]["bias"],
                            vgs[i - 1] if i > 0 else None,
                            vcs[i - 1] if i > 0 else None,
                            vgs[i], vcs[i], first=(i == 0))
      val_new["params"][h] = {"kernel": k2, "bias": b2}
    lk = np.asarray(val_p["hidden_2"]["kernel"])
    if lk.shape[0] != new_h[1]:
      lk = lk[vgs[1], :] / vcs[1][vgs[1]][:, None]
    val_new["params"]["hidden_2"] = {"kernel": lk,
                                        "bias": val_p["hidden_2"]["bias"]}
    params_new = [params[0], pol_new, val_new]
    cfg.network_factory_kwargs["policy_hidden_layer_sizes"] = tuple(new_h)
    cfg.network_factory_kwargs["value_hidden_layer_sizes"] = tuple(new_h)
    os.makedirs(new_root, exist_ok=True)
    checkpoint.save(new_root, 0, params_new, cfg,
                    config_fname="ppo_network_config.json")
    print("[widen] new checkpoint ->", os.path.join(new_root, "000000000000"))
    print("[widen] policy MLP:", [tuple(v["kernel"].shape) for v in pol_new["params"].values() if "kernel" in v])
    print("[widen] value  MLP:", [tuple(v["kernel"].shape) for v in val_new["params"].values() if "kernel" in v])


if __name__ == "__main__":
    main()
