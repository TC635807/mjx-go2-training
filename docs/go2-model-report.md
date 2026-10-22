# Go2 训练模型 / 场景 / 地形生成器 —— 交付报告 (M0, task-1)

> 交付人: asset-smith    日期: 2026-10-17
> 自检脚本 (Lead 提供): `sim/check_go2_model_contract.py` —— 两个场景都 **契约通过**。
> 环境: `/home/tc63/mujoco/.venv/bin/python` (mujoco 3.12.0 / jax 0.7.2 / warp-lang 1.16.0 / numpy 2.5.2 / scipy 1.18.1)。
> 所有命令都先 `export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15`。

---

## 1. 交付文件清单

| 文件 | sha256 (前 16) | 说明 |
|---|---|---|
| `models/go2/assets/` (18 个文件) | — | 16 个 Go2 mesh (`*.obj`, 从 `mjx-go2-mujoco2ros2/models/go2/assets` 原样 `cp -a`) + `terrain.png` + `terrain.json` |
| `models/go2/go2_mjx_position.xml` | `59ff5cc339db2961` | 机器人本体, 可被 include, 不含地面 |
| `models/go2/scene_mjx_position.xml` | `8816eaee5a7c6f73` | 平地场景 (`plane`) |
| `models/go2/scene_mjx_terrain.xml` | `9912e8c4efc8e07d` | hfield 地形场景 |
| `sim/gen_terrain_go2.py` | `d3d8b76b64b9f9ce` | 地形生成器 (从 `sim/gen_terrain.py` 派生) |
| `models/go2/assets/terrain.png` | `4d0cb7a6c1fa77ae` | 128×128 16bit 灰度高度场 |
| `docs/go2-model-report.md` | 本文件 | |

`terrain.png` 与 Go1 的 `models/go1/assets/terrain.png` **逐字节相同** (sha256 都是
`4d0cb7a6c1fa77aeb283114b36b4cac740a2755ce860289995f7be8623e8288c`) —— 默认参数一致时
生成器行为逐位继承, 这是"派生没有改坏东西"的最强证据。

---

## 2. 顺序契约 (最要紧的一条)

**训练模型的 body 树顺序与 actuator 顺序都被排成 `FR, FL, RR, RL` × (hip, thigh, calf)**
(部署基线 `go2.xml` 的 **body 文档顺序是 FL,FR,RL,RR**, 只有 actuator 顺序是 FR,FL,RR,RL)。
Lead 已确认此选择 (见 `envs/go2_walk.py` L60/L1006/L1009-1015/L570/L1628 的假设), 理由是
全项目冻结契约 `mjx-go2-mujoco2ros2/docs/interfaces.md §0` 与部署侧
`robot.py MOTOR_JOINTS` / `obs.py LEG` 都是 FR,FL,RR,RL。

实测顺序 (编译后用引擎打印):

```
train qpos 关节顺序 : ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint', 'FL_hip_joint', 'FL_thigh_joint',
                       'FL_calf_joint', 'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint', 'RL_hip_joint',
                       'RL_thigh_joint', 'RL_calf_joint']
train actuator 顺序 : ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
                       'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
deploy body 文档顺序: ['world', 'base_link', 'FL_hip', 'FL_thigh', 'FL_calf', 'FL_foot',
                       'FR_hip', 'FR_thigh', 'FR_calf', 'FR_foot', 'RL_hip', ... , 'RR_hip', ...]
deploy actuator 顺序: ['FR_hip', 'FR_thigh', 'FR_calf', 'FL_hip', 'FL_thigh', 'FL_calf',
                       'RR_hip', 'RR_thigh', 'RR_calf', 'RL_hip', 'RL_thigh', 'RL_calf']
```

因此 env 里 **action(ctrl 下标) / qpos[7:] / actuator_force 三者严格同序**, 不再需要任何重排。

---

## 3. 契约逐条对照

| task-1 要求 | 实现 | 证据 |
|---|---|---|
| 足端碰撞 geom 名 FL/FR/RL/RR 不变 | 保留 (id 20/32/44/56) | §4 |
| 新增 4 个 site, 名 FL/FR/RL/RR, `pos="-0.002 0 -0.213"` | 4 个 site 与同名足端 geom 同位 | §4 (`nsite=5` = imu + 4 足) |
| 躯干 body 名 `base_link`, imu site 名 `imu` | 保留 | §4 |
| sensor: gyro / local_linvel / accelerometer / global_linvel / global_angvel / upvector | 全部就位 (id 0..5) | §4 |
| sensor: `{F}_global_linvel` (framelinvel, objtype=site) ×4 | 就位 (id 6..9, dim=3) | §4 |
| sensor: `{F}_floor_found` (contact, geom1={F}, geom2=floor, reduce=mindist, num=1, data=found) ×4 | 就位 (id 10..13, dim=1) | §4 |
| 执行器 `<motor>` → `<position kp=20 kv=1 forcerange=...>` | 12 个 `<position kp=20 kv=1>`, gaintype=FIXED | §4 / §8 |
| ctrlrange = 关节限位 (不是力矩限位) | abduction ±1.0472 / front thigh [-1.5708,3.4907] / back thigh [-0.5236,4.5379] / knee [-2.7227,-0.83776] | §4 |
| forcerange = 力矩限位 (hip/thigh ±23.7, knee ±45.43) | 逐条一致 | §4 |
| 分类 abduction / front_hip / back_hip / knee, 前后腿 thigh 分开 | 4 个 default class, FR/FL 用 front_hip, RR/RL 用 back_hip | `go2_mjx_position.xml` |
| 足端 solimp=[0.9,0.95,0.023,0.5,2.0], condim=3 | 实测 condim=3, solimp=[0.9 0.95 0.023 0.5 2.0] | §4 |
| keyframe home: qpos z=0.32 + 12 关节角, ctrl 同值 | 见下 | §4 |
| 地形场景 floor geom 名 "floor" 且 type=hfield | 实测 `floor type=1 (mjGEOM_HFIELD)`, `nhfield=1` | §6 |
| hfield size 由生成器写回, 保证 h(原点)=0 且 h>=0 | `size="6 6 0.3000 0.0960"`, `h(0,0)=0.000000` | §6 |

keyframe (实测打印, qpos 与 ctrl 同值):

```
keyframe qpos = [ 0.    0.    0.32  1.    0.    0.    0.   -0.1   0.8  -1.5   0.1   0.8
                 -1.5  -0.1   1.   -1.5   0.1   1.   -1.5 ]
keyframe ctrl = [-0.1  0.8 -1.5  0.1  0.8 -1.5 -0.1  1.  -1.5  0.1  1.  -1.5]
qpos[7:] == ctrl : True
```

即 FR(-0.1, 0.8, -1.5) / FL(+0.1, 0.8, -1.5) / RR(-0.1, 1.0, -1.5) / RL(+0.1, 1.0, -1.5)。

---

## 4. 验证 1+2: 契约检查 + 沉降 (Lead 指定的自检脚本, 原始输出)

```
$ cd /home/tc63/mujoco/mjx-go2-training
$ /home/tc63/mujoco/.venv/bin/python sim/check_go2_model_contract.py
=== 契约检查: /home/tc63/mujoco/mjx-go2-training/models/go2/scene_mjx_position.xml ===
nq=19 nv=18 nu=12 nbody=18 ngeom=57 nsite=5 nsensor=14 nhfield=0
keyframe home: z=0.3200  joints=[-0.1  0.8 -1.5  0.1  0.8 -1.5 -0.1  1.  -1.5  0.1  1.  -1.5]
                ctrl  =[-0.1  0.8 -1.5  0.1  0.8 -1.5 -0.1  1.  -1.5  0.1  1.  -1.5]

执行器:
  FR_hip     joint=FR_hip_joint     jnt=[-1.047  1.047] ctrl=[-1.047  1.047] forcerange=[-23.7  23.7]
  FR_thigh   joint=FR_thigh_joint   jnt=[-1.571  3.491] ctrl=[-1.571  3.491] forcerange=[-23.7  23.7]
  FR_calf    joint=FR_calf_joint    jnt=[-2.723 -0.838] ctrl=[-2.723 -0.838] forcerange=[-45.43  45.43]
  FL_hip     joint=FL_hip_joint     jnt=[-1.047  1.047] ctrl=[-1.047  1.047] forcerange=[-23.7  23.7]
  FL_thigh   joint=FL_thigh_joint   jnt=[-1.571  3.491] ctrl=[-1.571  3.491] forcerange=[-23.7  23.7]
  FL_calf    joint=FL_calf_joint    jnt=[-2.723 -0.838] ctrl=[-2.723 -0.838] forcerange=[-45.43  45.43]
  RR_hip     joint=RR_hip_joint     jnt=[-1.047  1.047] ctrl=[-1.047  1.047] forcerange=[-23.7  23.7]
  RR_thigh   joint=RR_thigh_joint   jnt=[-0.524  4.538] ctrl=[-0.524  4.538] forcerange=[-23.7  23.7]
  RR_calf    joint=RR_calf_joint    jnt=[-2.723 -0.838] ctrl=[-2.723 -0.838] forcerange=[-45.43  45.43]
  RL_hip     joint=RL_hip_joint     jnt=[-1.047  1.047] ctrl=[-1.047  1.047] forcerange=[-23.7  23.7]
  RL_thigh   joint=RL_thigh_joint   jnt=[-0.524  4.538] ctrl=[-0.524  4.538] forcerange=[-23.7  23.7]
  RL_calf    joint=RL_calf_joint    jnt=[-2.723 -0.838] ctrl=[-2.723 -0.838] forcerange=[-45.43  45.43]

足端接触:
  FR: condim=3 solimp=[0.9   0.95  0.023 0.5   2.   ] friction=[0.4  0.02 0.01] prio=1
  FL: condim=3 solimp=[0.9   0.95  0.023 0.5   2.   ] friction=[0.4  0.02 0.01] prio=1
  RR: condim=3 solimp=[0.9   0.95  0.023 0.5   2.   ] friction=[0.4  0.02 0.01] prio=1
  RL: condim=3 solimp=[0.9   0.95  0.023 0.5   2.   ] friction=[0.4  0.02 0.01] prio=1

=== 沉降测试 (home keyframe, 1s) ===
  躯干 z = 0.2662 m   倾角 = 0.51°   ncon = 4   |tau|max = 6.89 N·m

=== 结论 ===
  契约通过。
CONTRACT_FLAT_EXIT=0
```

```
$ /home/tc63/mujoco/.venv/bin/python sim/check_go2_model_contract.py --xml models/go2/scene_mjx_terrain.xml
=== 契约检查: models/go2/scene_mjx_terrain.xml ===
nq=19 nv=18 nu=12 nbody=18 ngeom=57 nsite=5 nsensor=14 nhfield=1
keyframe home: z=0.3200  joints=[-0.1  0.8 -1.5  0.1  0.8 -1.5 -0.1  1.  -1.5  0.1  1.  -1.5]
                ctrl  =[-0.1  0.8 -1.5  0.1  0.8 -1.5 -0.1  1.  -1.5  0.1  1.  -1.5]

执行器: (与平地场景逐字相同, 略)
足端接触: (与平地场景逐字相同, 略)

=== 沉降测试 (home keyframe, 1s) ===
  躯干 z = 0.2658 m   倾角 = 0.70°   ncon = 6   |tau|max = 7.07 N·m

=== 结论 ===
  [warn] ncon=6 != 4 (可能不是四足均匀着地)
  契约通过。
CONTRACT_TERR_EXIT=0
```

> 地形场景 ncon=6 是**正常现象**: hfield 与足端球在一个格子里会给出 1~2 个接触点
> (平地 plane 是解析平面, 恒为 4)。`[warn]` 不是 `[FAIL]`。

平地沉降的完整时间线 (自己写的探针, 采样 0.2s):

```
=== [1] 平地 plane: home keyframe 沉降 0.2s 采样 ===
  t=0.000s z=0.3200
  t=0.2s z=0.2724 倾角= 2.71° ncon=4
  t=0.4s z=0.2707 倾角= 1.51° ncon=4
  t=0.6s z=0.2696 倾角= 0.01° ncon=4
  t=0.8s z=0.2665 倾角= 0.49° ncon=4
  t=1.0s z=0.2662 倾角= 0.51° ncon=4
  1s 后足端-地面接触:
    floor -FR     dist=-0.00266
    floor -FL     dist=-0.00266
    floor -RR     dist=-0.00312
    floor -RL     dist=-0.00312
```

**期望 z≈0.26-0.28, ncon=4 —— 实测 z=0.2662, ncon=4, 四足全部着地。**
(Lead 之前手测 0.268/1.58° 与这里的 0.2662/0.51° 同档, 差异来自求解器设置与
`margin` 清零, 见 §7。)

---

## 5. 验证 4: warp 冒烟 (GPU)

### 5.1 Lead 指定的那条命令 (本次失败点)

```
$ XLA_PYTHON_CLIENT_MEM_FRACTION=0.15 /home/tc63/mujoco/.venv/bin/python -c "
import mujoco, mujoco.mjx as mjx
m=mujoco.MjModel.from_xml_path('models/go2/scene_mjx_terrain.xml')
mw=mjx.put_model(m, impl='warp'); print('warp put_model OK', mw.nv)
"
.../mujoco_warp/_src/io.py:705: UserWarning: MULTICCD is enabled, but the scene contains CCD pairs
  without multicontact support: [('HFIELD', 'CYLINDER'), ('HFIELD', 'BOX'), ('CYLINDER', 'CYLINDER'),
  ('CYLINDER', 'BOX')]. At most 1 contact will be generated for these pairs.
warp put_model OK 18
PUT_MODEL_EXIT=0
```

### 5.2 20 步 jit 前向 (平地)

```
[flat] host-load+put_model+make_data 1.92s | jit 编译+首步 163.29s | 其余 19 步 1.00s (52.5 ms/步)
[flat] warp: 20 步后 base z = 0.3136 m   qpos finite = True   qacc finite = True
[flat] ctrl finite = True   总耗时 166.40s
```

### 5.3 20 步 jit 前向 (地形, hfield)

```
[terrain] host-load+put_model+make_data 2.05s | jit 编译+首步 2.98s | 其余 19 步 0.88s (46.5 ms/步)
[terrain] warp: 20 步后 base z = 0.3137 m   qpos finite = True   qacc finite = True
[terrain] ctrl finite = True   总耗时 6.19s
```

要点:
- 两个场景都 `finite=True` (qpos/qacc/ctrl), base z 从 0.32 落到 0.3136/0.3137
  (20 步 = 0.04s, 还在下落中, 合理)。
- **首次 warp 编译很贵 (~163s)**, 且几乎全部花在 4 个 CCD kernel 上
  (`ccd_kernel_... 26889ms / 28907ms / 52360ms / 50612ms`)。warp kernel cache
  写进 `/home/tc63/.cache/warp/1.16.0`, **后续进程 2.98s 就编译完** (5.3 的实测)。
  所以第一次训练启动会卡 2-3 分钟, 不是挂死。这是本环境 (warp 1.16.0 + sm_120) 的客观开销。
- 冒烟脚本用 `mjx.make_data(m, impl="warp", naconmax=2048, njmax=256)` ——
  **第一个参数是 `mujoco.MjModel`** (不是 mjx Model), 与 task 描述一致。

---

## 6. 验证 3: 地形场景 hfield 与坐标约定

```
nhfield=1  floor geom: id=0 type=1 (mjGEOM_HFIELD=1) hfieldid=0 name='terrain' nrow=128 ncol=128
size=[6.  6.  0.3  0.096]
elevation_z=0.3  base_z=0.096  (data 形状 (128, 128))

[A] mj_ray 高度 vs env 的双线性直读 (h = bilinear(data)*elevation_z)
   (x,y)        mj_ray     直读*elev    差
  (+0.00,+0.00)    0.00000    0.00000  +0.000000
  (+1.00,-3.82)    0.06000    0.06000  +0.000000
  (-3.82,+1.00)    0.06000    0.06000  +0.000000
  (+3.00,+3.00)    0.03529    0.03507  +0.000221
  (-5.50,-5.50)    0.00000    0.00000  +0.000000
  (+0.50,-4.80)    0.00000    0.00000  +0.000000
  最大偏差 = 0.000221 m

[B] 原点 h(0,0) (reset 落点)
  mj_ray h(0,0) = 0.000000 m   命中 geom id 0 (floor=0)
  data 中心格 = 0.000000 * elev = 0.000000
  min/max/mean(直读*elev) = 0.0000 / 0.3000 / 0.0343 m

=== [2] 地形场景: 原点沉降 (pos.z=0 标定复验) ===
  地形 hfield 1s: z=0.2658 倾角=0.70° ncon=6
  平地 plane  1s: z=0.2662 ncon=4  -> Δz=0.42 mm
  floor_found (FR,FL,RR,RL) = [1.0, 2.0, 2.0, 1.0]
  {F}_global_linvel (FR,FL,RR,RL) = [[0.0002,-0.0002,-0.0003], [0.0002,0.0001,-0.0001],
                                     [0.0001,-0.0002,0.0003], [0.0001,0.0002,0.0005]]
  global_linvel/upvector = [0.0017,-0.0001,0.0009] [-0.0119,-0.0017,0.9999]
```

- `nhfield=1` ✓, floor 是 hfield ✓, `h(0,0)=0.000000` ✓, 起伏只向上 (min=0) ✓。
- 坐标约定**继承**生成器的结论并复验通过: 设计上"沿 +x 的楼梯在第 3 象限 (y≈-3.8)"
  和"沿 +y 的楼梯在第 4 象限 (x≈-3.8)"在引擎里都落在高度 0.0600 m (第 1 级踏步面),
  说明 `png = flipud(h.T)` 的修正对 Go2 同样成立 (世界(x,y) == 生成器(x,y))。
- `mj_ray` 与 env 的双线性直读最大差 0.22 mm (Go1 报告的是 0.35 mm), 量级一致:
  差异来自格点间三角形平面的双线性近似, 不影响 reset 对齐。
- `base_z=0.096` **不影响地表高度** (mj_ray 与 `data*elevation_z` 逐点吻合), 只撑几何包围盒;
  与 Go1 的做法完全一致 (`terrain.json` 里记的 `base_z: 0.0` 是生成器语义值, XML 里必须是正的 0.096)。
- 地形场景原点沉降 z=0.2658 vs 平地 0.2662 → **Δ=0.42 mm**, 证明 `floor pos.z=0` 的标定成立
  (因为 flatten_origin 保证原点被抹平, reset 正落在原点)。

地形场景的行为边界 (CPU, `njmax` 动态分配):

```
=== [3] 地形上保持 home 姿态 6s (稳定性) ===
  6s: z=0.2665 倾角=0.47° ncon=6 finite=True 漂移=71.6 mm

=== [4] 6s 随机位置命令 / 6s 全量程正弦 (NaN + 约束溢出) ===
  随机 6s / 平地: qpos finite=True qvel finite=True sensordata finite=True ncon_max=7 溢出=0
  随机 6s / 地形: qpos finite=True qvel finite=True sensordata finite=True ncon_max=16 溢出=0
  正弦 6s / 平地: qpos finite=True qvel finite=True sensordata finite=True ncon_max=13 溢出=0
  正弦 6s / 地形: qpos finite=True qvel finite=True sensordata finite=True ncon_max=28 溢出=0
```

没有 NaN, 没有约束溢出 (`mjWARN_CNSTRFULL` 计数 0); ncon 峰值 28 (Go1 同款地形报告的是 11,
Go2 更大, 因为足端球半径 2.2cm 与格距 9.4cm 的比例不同 —— warp 侧 `njmax=256` 有富余)。

---

## 7. 与部署基线 `go2.xml` 的逐项 parity (按名字, 非文档顺序)

用 `mujoco` 把两个模型都加载后按**名字**逐项比对:

```
=== 1) body 质量/惯量 (按名字) ===
  body 数=18  质量/惯量不一致的 body: 0
  总质量 train=15.206408 kg  deploy=15.206408 kg

=== 2) 关节参数 (按名字) ===
  关节数=12  有差异的=0            # range / damping / armature / frictionloss / stiffness 全同

=== 3) 碰撞几何 (按 body+类型+尺寸 匹配, 排除场景 floor) ===
  参与碰撞的 geom: train=23 deploy=23
  不一致的 body 数=0               # 类型/尺寸/pos/quat 全同

=== 4) 关键接触参数 (按 geom 名) ===
  FR: train  pos=[-0.002 0. -0.213] size=[0.022 0. 0.] condim=3 solimp=[0.9 0.95 0.023 0.5 2.] fric=[0.4 0.02 0.01] prio=1 margin=0.0
       deploy pos=[-0.002 0. -0.213] size=[0.022 0. 0.] condim=6 solimp=[0.9 0.95 0.001 0.5 2.] fric=[0.4 0.02 0.01] prio=1 margin=0.001
  (FL/RR/RL 同样)

=== 6) option ===
  train : cone=1(elliptic) impratio=100.0 timestep=0.002 iters=100 disableflags=0 margin_unique=[0.]
  deploy: cone=1(elliptic) impratio=100.0 timestep=0.002 iters=100 disableflags=0 margin_unique=[0.001]
```

**结论: 除下面 §8 列出的 4 条刻意差异外, 训练模型与部署基线在名字维度上一项不差。**

---

## 8. 与部署基线的差异 / 未满足项 (全部主动披露)

1. **四条腿的 body 树顺序**: 训练模型 = `FR,FL,RR,RL`, 部署基线文档顺序 = `FL,FR,RL,RR`。
   纯排列, 物理参数逐项相同 (§7)。**原因**: 全项目冻结契约与 env 的
   `FEET`/`qpos[7:].reshape(4,3)`/`trot_phase=[0,π,π,0]`/`sym_ema` 块索引都是 FR,FL,RR,RL;
   部署侧 `MOTOR_JOINTS` 也是这个顺序, 所以策略的 obs/action 顺序在两边一致。
   **这是 task-1 描述里"预期 FL,FR,RL,RR"那句的唯一偏离, Lead 已确认。**

2. **`margin` 全部从 0.001 改为 0** (唯一的数值差异)。
   基线默认 geom 类带 `margin="0.001"`, 而 mujoco_warp 在 MULTICCD 打开时拒绝任何带非零
   margin 的 CCD 接触对, `mjx.put_model(impl="warp")` 直接抛
   `NotImplementedError: geom pair (6, 14) has non-zero margin ((0.001, 0.001)) with MULTICCD enabled`,
   训练起不来 (Go1 训练模型根本没有 margin, 所以从没踩到)。
   实测三种修法: (a) `geom_margin[:]=0` → OK; (b) 只 disable MULTICCD → 仍报 NATIVECCD,
   必须同时 disable NATIVECCD 才过 (要牺牲 CCD, 影响更大); (c) 只把 BOX/MESH 的 margin 置 0 → OK。
   **采用 Lead 指定的 (a): 统一清零**, 与 Go1 训练模型一致。
   影响: 1mm 以内的接触生成时机; 平地沉降 z 由 0.2672 → 0.2662 (Δ=1.0 mm), 倾角 0.49°→0.51°。
   **部署模型 `go2.xml` 未改**, parity 检查请按"训练 margin=0 / 部署 margin=0.001"这个口径对比。

3. **足端接触由硬接触改为软接触** (task 要求): `condim=3`, `solimp=[0.9,0.95,0.023,0.5,2.0]`
   (部署是 `condim=6` + 默认硬 solimp)。env 的 `foot_soft_contact=False` 可在运行时改回硬接触。

4. **新增 sensor 段与 4 个足端 site** (task 要求, 部署模型没有)。

其它说明 (不是问题, 但会影响对比脚本):
- 训练 XML 里 `base_link` 下保留了基线的 4 个空 body (`{F}_foot`, 无 geom/无 joint) 以求同源;
  它们不产生 DOF 也不参与碰撞。
- 训练模型多一个 `floor` geom (在场景文件里), 所以 `ngeom=57` vs 部署单独加载时的 56。
- `terrain.json` 里 `base_z: 0.0` 是生成器语义值; XML 里 `size` 第 4 项必须是正的
  `0.096` (MuJoCo 拒绝 0), 两者不矛盾 (Go1 同款)。

---

## 9. 地形生成器 `sim/gen_terrain_go2.py`

从 `sim/gen_terrain.py` 派生。`diff` 全文只有 89 行, 且全是下面这些非算法改动:
- 输出路径 `models/go1/assets` → `models/go2/assets`, 同步目标
  `models/go1/scene_mjx_terrain.xml` → `models/go2/scene_mjx_terrain.xml`;
- 把原本 `argparse` 解析了却**没接上**的 `--flat_radius/--flat_blend` 接到 `make_terrain`
  (默认 0.5/0.9 与原硬编码相同 → 默认档行为逐位不变);
- 文档串/注释里的 Go1→Go2、路径、足端半径 2.3→2.2cm; `terrain.json` 多一个 `"model": "go2"` 字段;
- 变量名 `stair_info` → `flights_info` (原变量赋值后未使用)。

`make_terrain / smooth_noise / add_rect_step / flatten_origin / add_staircase / save / sync_xml`
的算法**一行未改**, 坐标约定 (`png = flipud(h.T)`) 原样继承 (见 §6 复验)。

默认档 (n=128, ext=6.0, bump_amp=0.05, sigma=2.0, step_height=0.06, stair_steps=5,
n_staircases=2, stair_pitch=0.55, stair_ramp=2):

```
$ /home/tc63/mujoco/.venv/bin/python sim/gen_terrain_go2.py
地形参数: 128x128  范围 ±6.0m  格距 9.4cm
  起伏幅度=0.05m (sigma=2.0)  台阶高=0.06m  台地数=6  阶梯=2条×5级  seed=0
  高度: min=0.0000 max=0.3000 mean=0.0346 m
  原点高度 h(0,0)=0.000000 m  (必须≈0 否则 reset 落地不一致)
  最大坡度: x=29.8° y=29.8°  (足端半径 2.2cm vs 格距 9.4cm)

已写: /home/tc63/mujoco/mjx-go2-training/sim/../models/go2/assets/terrain.png
已写: /home/tc63/mujoco/mjx-go2-training/sim/../models/go2/assets/terrain.json
  elevation_z=0.3000 (= max height, 用于 XML size)
已同步 elevation_z -> scene_mjx_terrain.xml (hfield size)
```

---

## 10. 复现命令全集

```bash
cd /home/tc63/mujoco/mjx-go2-training
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.15
PY=/home/tc63/mujoco/.venv/bin/python

# 1) 重新生成地形 (+ 自动同步 hfield size)
$PY sim/gen_terrain_go2.py

# 2) 契约检查 (平地 / 地形)
$PY sim/check_go2_model_contract.py
$PY sim/check_go2_model_contract.py --xml models/go2/scene_mjx_terrain.xml

# 3) warp put_model (训练启动的失败点)
$PY -c "
import mujoco, mujoco.mjx as mjx
m=mujoco.MjModel.from_xml_path('models/go2/scene_mjx_terrain.xml')
mw=mjx.put_model(m, impl='warp'); print('warp put_model OK', mw.nv)"

# 4) warp 20 步冒烟 (平地替换 xml 即可; 首次编译 ~163s, 之后 ~3s)
$PY - <<'EOF'
import numpy as np, jax, jax.numpy as jp, mujoco
from mujoco import mjx
m = mujoco.MjModel.from_xml_path('models/go2/scene_mjx_position.xml'); m.opt.timestep = 0.002
mx = mjx.put_model(m, impl='warp')
kf = m.keyframe('home')
d = mjx.make_data(m, impl='warp', naconmax=2048, njmax=256)
d = d.replace(qpos=jp.asarray(kf.qpos, dtype=jp.float32), ctrl=jp.asarray(kf.ctrl, dtype=jp.float32))
step = jax.jit(lambda mx, d: mjx.step(mx, d))
for _ in range(20): d = step(mx, d)
q = np.asarray(jax.block_until_ready(d.qpos))
print('base z =', float(q[2]), 'finite =', bool(np.all(np.isfinite(q))))
EOF
```
