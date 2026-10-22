import os,sys
sys.path.insert(0,'/home/tc63/mujoco/mjx-go2-training')
import jax, jax.numpy as jp, numpy as np
from envs.go2_walk import Go2Walk, default_config
for scene in ['models/go2/scene_mjx_terrain_mix.xml','models/go2/scene_mjx_terrain_v5.xml']:
    cfg=default_config(); cfg.terrain=True; cfg.terrain_xml=scene
    hs=cfg.height_scan; hs.enable=True; hs.nx=17; hs.ny=11
    hs.x_min=-0.4; hs.x_max=1.2; hs.y_min=-0.5; hs.y_max=0.5
    cfg.random_init.enable=True; cfg.naconmax=32*16; cfg.obs_noise.enable=True
    env=Go2Walk(config=cfg)
    N=4
    st=jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0),N))
    obs=np.asarray(st.obs["state"]); priv=np.asarray(st.obs["privileged_state"])
    st2=jax.vmap(env.step)(st, jp.zeros((N, env.mjx_model.nu)))
    print(scene.split('/')[-1], '| scan=%dx%d=%d'%(hs.nx,hs.ny,hs.nx*hs.ny),
          '| actor', obs.shape, 'priv', priv.shape,
          '| scan feature range', round(float(obs[0,48:83].min()),3) if obs.shape[1]>=83 else None,
          '| finite', bool(np.isfinite(np.asarray(st2.obs["state"])).all()))
