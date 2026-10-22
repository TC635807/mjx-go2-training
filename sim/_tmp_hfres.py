import os, sys, numpy as np, mujoco
sys.path.insert(0, '/home/tc63/mujoco/mjx-go2-training/sim')
from gen_terrain_go2_mix import build, SCENE_TMPL, write_png

ASSETS = '/home/tc63/mujoco/mjx-go2-training/models/go2/assets'
SCENE = '/home/tc63/mujoco/mjx-go2-training/models/go2'
msgs = []
try:
    mujoco.set_mju_user_warning(lambda msg: msgs.append(str(msg)))
    print("warning callback set")
except Exception as e:
    print("no callback:", e)

def make_scene(n):
    H, meta = build(n, 6.0, "mix", 1)
    elev = float(np.ceil(H.max() * 20) / 20); elev = max(elev, 0.05)
    prefix = "terrain_hftest_%d" % n
    write_png(H, elev, os.path.join(ASSETS, prefix + ".png"))
    path = os.path.join(SCENE, "scene_hftest_%d.xml" % n)
    meta.update(elevation=elev, cell_m=12.0 / n, prefix=prefix)
    open(path, "w", encoding="utf-8").write(SCENE_TMPL.format(**meta, elev=elev))
    return path, H

def run(path, n_ep=8, steps=700, seed=0):
    m = mujoco.MjModel.from_xml_path(path); m.opt.timestep = 0.002
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    rng = np.random.default_rng(seed)
    nwarn = 0; nconmax = 0
    for ep in range(n_ep):
        d.qpos[:] = m.key_qpos[key].copy(); d.qvel[:] = 0; d.ctrl[:] = m.key_ctrl[key]
        mujoco.mj_forward(m, d)
        # 随机出生点 (任意地形上) + 随机朝向 + 随机初始速度
        xy = rng.uniform(-5.0, 5.0, 2)
        gg = np.array([1,0,0,0,0,0], dtype=np.uint8); gid = np.zeros(1, dtype=np.int32)
        dist = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], 2.0]), np.array([0.,0.,-1.]), gg, 1, -1, gid)
        h = 2.0 - dist if dist > 0 else 0.0
        yaw = rng.uniform(-np.pi, np.pi); q = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
        d.qpos[:3] = [xy[0], xy[1], h + 0.34]; d.qpos[3:7] = q
        d.qvel[:6] = rng.uniform(-0.5, 0.5, 6)
        mujoco.mj_forward(m, d)
        before = len(msgs)
        for i in range(steps):
            if i % 10 == 0:
                d.ctrl[:] = m.key_ctrl[key] + rng.uniform(-1.2, 1.2, m.nu)
            mujoco.mj_step(m, d)
            if not np.all(np.isfinite(d.qpos)): break
        nwarn += len(msgs) - before
    return nwarn, len(msgs)

for n in (256, 192, 160, 128):
    path, H = make_scene(n)
    nw, tot = run(path)
    print("n=%3d (格距 %.2fcm): 警告 %d 次 (总消息 %d)" % (n, 120.0/n/10, nw, tot))
