# Go2 激光雷达遮挡/盲区 vs 训练侧"完美高度场" —— sim2real gap 技术调研

> 任务：调研真实 Go2 激光雷达/高程图的遮挡、盲区、空洞问题，以及开源四足 RL 与高程图领域如何建模/处理，给出可落地的训练侧改进建议。
> 方法：纯网络调研（官方规格页、开源仓库源码、论文），**未做任何实机测量、未训练、未改代码**。
> 调研时间：2026-10。
> 与本仓库的关系：当前 [mjx-go1-getup](mjx-go1-getup) 的 height_scan 是对**解析高度场**直接双线性采样（无限分辨率、无遮挡、无 FOV、无 miss），而真机上这是"高程图 + 点云 + 位姿估计"的多级带噪管线。本文档第 5 节给出把这条 gap 补上的具体参数方案。

---

## 0. 结论先行（TL;DR）

1. **Go2 标配的 Unitree 4D LiDAR L1 官方指标是 360°×90°、最小测距 0.05 m**；但"最小测距小"不等于"脚下地面可见"——近场可见性由**垂直 FOV 的下向角度、安装高度、机体/腿自遮挡**三者共同决定，官方没有给出 L1 垂直 FOV 的角度上下界（我们未能核实，见附录 B）。
2. **URDF（unitree_ros go2_description）给出 radar 关节在 base 系 xyz=(0.28945, 0, -0.046825)、rpy=(0, 2.8782 rad, 0)**，即雷达安装在机体**前向 0.29 m、比 base 原点低 4.7 cm**的位置，且相对 L1 SDK 坐标系**倒置约 180°、再俯仰偏 15°**（该姿态的物理含义需实机点云核对）。按训练站高 0.32 m 估算，射线原点离地约 **0.27 m**；若按顶部安装假设则约 0.35–0.40 m，两种情况下近场盲区半径约 **0.16–0.55 m**（见 §1.3 表）。
3. **legged_gym / IsaacLab / mjlab 的 height scan 都不模拟机体自遮挡，也不模拟 FOV 盲区**：legged_gym 直接查高度场（_get_heights），IsaacLab RayCaster 只对**静态地形 mesh** 打射线，mjlab terrain_scan 显式 include_geom_groups=(0,)（只打地形）。三者都只做"加性噪声 + 位姿漂移（drift）"级别的退化。**没有一家建模"未观测格点"。**
4. **mjlab 的 terrain_scan 参数与我们几乎同构**：GridPatternCfg(size=(1.6, 1.0), resolution=0.1) → 17×11=187 条射线、ray_alignment="yaw"、max_distance=5.0、actor 高度噪声 ±0.1 m、critic 无噪——我们的 17×11 网格就是这套几何（只是把 x 范围前移了）。其 miss 处理是 miss_value 默认取 max_distance（常量填充），IsaacLab 则直接把 miss 记为 inf（未处理）。
5. **高程图一侧（elevation_mapping_cupy / ANYmal 系）已经有一套成熟的遮挡与空值处理**：enable_visibility_cleanup 用射线清理被遮挡单元（max_ray_length=10 m, cleanup_step=0.1 m），min_valid_distance=0.1 m 丢弃近距点，方差随时间增长（time_variance / max_variance）实现"衰减"，空值用 min_filter 邻域最小值 / inpaint（OpenCV）填充；地图以 base_link 为中心（robot-centric）而非世界系。
6. **鲁棒训练的主流答案是"非对称 AC + 特权 critic + 感知退化随机化 + 回退本体感知"**：Miki 2022 的 belief encoder 在遮挡/高反/雪地时"无缝退回本体感知"；DreamWaQ/DreamWaQ++ 用置信 mask 丢掉不可靠点特征并保留盲走 backbone；RENet 用双估计器（视觉/本体）+ 逐环境 mask 切换；Beyond Egocentric 明确做"相机 dropout"；DPL 做"自遮挡感知的 ray casting"；2026 年 MDPI 的 ray-cone 遮挡增强报告结构化扇区缺失下补全误差降低 8.3–9.7%。**未找到**把"未观测格点统一填 0 + mask 通道"作为 legged RL 标准做法的公开工作——这正是我们可以自己定义的空间。
7. **建议方向（详见 §5）**：保持 actor 输入 187 维高度特征不变，新增 **FOV+几何自遮挡 mask（机体系静态预计算）+ 每步随机丢点/扇区遮挡 + 高度噪声加大到 0.05 m + 全帧失明 10–20%**，未观测格点填 **0（中性，等价"假设平地"）** 并附带 **8–12 维扇区可见性/置信特征**；critic 保留 clean 高度场（现有非对称 AC 不动）；遮挡强度与地形课程联动、从 0 递增。
8. 关键验收：在 eval 里做 **height scan 消融**（clean vs 遮挡 vs 全 0），若全 0 时回报不下降说明策略已退化成盲走，可用"遮挡下回报下降 < X% 但盲走下降 > Y%"作为鲁棒性指标。

---

## 1. Go2 的激光雷达：型号、安装、视场、近场

### 1.1 型号与官方规格

Go2 官方产品页把宇树自研 **Unitree 4D LiDAR L1** 列为标配（"Standard Ultra-wide 4D LIDAR"），原文：

> "Go2 features with Unitree's self-developed 4D LiDAR L1 with **360°x90° hemispherical** ultra-wide recognition, **super small blind spot** and a **minimum detection distance as low as 0.05m**."
> —— <https://www.unitree-robot.com/go2/>

L1 官方产品参数页（中文站，含 L1 PM / L1 RM 两个型号）：

| 参数 | L1 PM | L1 RM |
|---|---|---|
| FOV | **360° × 90°**（全向超广角、非重复扫描） | 同左 |
| 扫描距离 | 20 m @90% 反射率 / 10 m @10% | 30 m @90% / 15 m @10% |
| 近处盲区 | **0.05 m** | 0.05 m |
| 采样频率 / 有效频率 | 43200 点/s / 21600 点/s | 同左 |
| 周向扫描频率（水平） | **11 Hz** | 同左 |
| 竖直扫描频率 | **180 Hz** | 同左 |
| 体积 / 重量 | 75×75×65 mm / 230 g | 同左 |
| IMU | 3 轴加速度 + 3 轴陀螺（250 Hz 推送） | 同左 |
| 扫描方式 | 非接触无刷转镜、非重复扫描 | 同左 |

来源：<https://www.unitree.com/cn/mobile/LiDAR>（参数对比表与"产品型号"表）；同样的参数也见百度百科条目 <https://baike.baidu.com/item/Unitree%204D%20LiDAR%20L1/62957619>（引用了官网）。拆机分析确认 L1 是"高低速双电机 + 转镜"的机械式扫描，垂直 90°、水平 360°，且"越靠近 FOV 中心点云密度越大"：<https://developer.horizon.auto/blog/13003>（转载：<https://www.gansuan.com/tech/sharing/2776029949176213504.html>）。

补充说明（可选装/改装，非 Go2 出厂标配）：

- **Livox Mid-360** 官方指标为 **360°×59°**、**最小测距 0.1 m**、40 m @10% 反射率：<https://www.livoxtech.com/mid-360>。垂直 59° 的上下界（业界资料常引 -7°~+52°）我们**未从可访问的官方文件直接核实**；官方用户手册 PDF 链接为 <https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Mid360/Livox_Mid-360_User_Manual_EN.pdf>（本次工具无法解析 PDF 正文）。Go2 装 Mid-360 属于社区改装（有第三方安装支架在售，如 <https://cn.made-in-china.com/gongying/hylan2025-FeIQSzRCZpHD.html>），不是原厂配置。

> 结论：**出厂配置下，"脚下/近场地面能否被看到"不能只看 0.05 m 这个数**——它描述的是"测距模块对近物的响应能力"，不是"地面被扫到"。真正的判据是垂直 FOV 的下向角 + 安装高度 + 自遮挡（§1.3）。

### 1.2 安装位置（机体系坐标 / 离地高度 / 是否前倾）

**官方 URDF 给出的雷达位姿**（unitree_ros 的 Go2 描述文件，radar 关节）：

```xml
<joint name="radar_joint" type="fixed">
  <origin xyz="0.28945 0 -0.046825" rpy="0 2.8782 0" />
  <parent link="base" />
  <child link="radar" />
</joint>
```

来源：<https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf>（社区 ROS2 驱动里的同一组数值：<https://github.com/abizovnuralem/go2_ros2_sdk/blob/master/go2_robot_sdk/urdf/go2.urdf>）。

解读与不确定度：

- **x = +0.28945 m**：雷达在机体坐标系前向约 0.29 m（机体几何中心之前，靠近"头部"前缘；URDF 中 Head_upper 关节在 x=0.285、z=+0.01，front_camera 在 x=0.327、z=+0.043）。
- **z = -0.046825 m**：相对 base 原点**低 4.7 cm**。
- **rpy pitch = 2.8782 rad ≈ 164.9° = 180° − 15.1°**：相对 L1 自身 SDK 坐标系（原点在**底部安装面中心**，+Z 垂直向上，+X 与出线方向相反，见 <https://github.com/unitreerobotics/unilidar_sdk> README 的 "Coordinate System Definition"）等价于**倒置约 180°、再俯仰偏 15°**。这到底是"倒装 + 前倾 15°"的物理事实，还是 URDF 对 Unitree 内部 radar 坐标约定的建模，我们**未找到官方文字说明**，建议实机用点云对地面/墙面成像核对（把点云按该 TF 转到 base 系，看地面平面是否水平、正前方是否为 +X）。
- **离地高度**：本仓库 Go2 训练模型的 home keyframe 站高为 **z=0.32 m**（[models/go2/go2_mjx_position.xml](mjx-go1-getup/models/go2/go2_mjx_position.xml)，与 [envs/go2_walk.py](mjx-go1-getup/envs/go2_walk.py) 的 spawn_z_offset=0.32 一致），按 URDF 相对高度算，**雷达射线原点离地约 0.32−0.047 ≈ 0.27 m**。但如果实机是"顶部安装"（Go2 外观上 L1 位于头部上方），则离地约 0.35–0.40 m。**两种假设都保留**，在 §5 用参数 lidar_z 暴露出来、实机标定。
- 另有一个官方静态 TF 事实：Go2 的 ROS2 驱动发 radar -> utlidar_lidar 静态 TF（无平移参数），社区教程里也确认这条链存在：<https://ztl3106742440-hub.github.io/go2-tutorial/02-packages/06-driver/>。

> 未找到项：Go2 官方安装图纸/CAD（给出雷达离地高度与倾角的文字说明）；L1 垂直 FOV 的角度上下界；实机点云的自遮挡实测统计。见附录 B。

### 1.3 "脚下/近场地面能不能看到"的几何分析（本文档的推导，非实测）

设雷达射线原点离地 h，垂直 FOV 的最大下俯角 θ_down，则**平坦地面上**能被看到的最小水平半径

```
r_min = h / tan(θ_down)          # 近场几何盲区半径
```

另外还有两条硬约束：L1 最小测距 0.05 m（<https://www.unitree-robot.com/go2/>）；高程图侧还会丢弃近距点（elevation_mapping_cupy 的 min_valid_distance: 0.1，见 §3）。

| 雷达离地 h | θ_down=30° | θ_down=45° | θ_down=60° |
|---|---|---|---|
| 0.27 m（URDF 口径） | r_min ≈ 0.47 m | **0.27 m** | 0.16 m |
| 0.32 m | 0.55 m | 0.32 m | 0.18 m |
| 0.40 m（顶部安装口径） | 0.69 m | 0.40 m | 0.23 m |

把 r_min、近距截断（0.10 m）、以及**躯干盒自遮挡**（Go2 URDF base collision box：0.3762×0.0935×0.114 m，中心在 base，即 x∈[-0.188,0.188]、y∈[-0.047,0.047]、z∈[-0.057,0.057]）三者叠加，对我们 17×11 网格（x∈[-0.4,1.2], y∈[-0.5,0.5], 间距 0.1 m）做近似计数（仅躯干、未含腿/头，故是**下界**）：

| 雷达离地 | θ_down=30° | θ_down=45° | θ_down=60° |
|---|---|---|---|
| 0.273 m | 69/187 不可见（37%） | 21/187（11%） | 9/187（5%） |
| 0.32 m | 104/187（56%） | 42/187（22%） | 16/187（9%） |
| 0.40 m | 138/187（74%） | 65/187（35%） | 39/187（21%） |

（估算脚本的假设与复现方式见附录 A。仅加腿/头/台阶边缘遮挡后，不可见比例只会更高。）

对我们网格的另一条重要事实：**x<0.19 m 的格子有 66 个（35%）**，它们位于躯干前缘之后；从"前上方雷达"看，这些格子的视线大概率穿过躯干（若雷达低到 0.27 m 则射线从躯干下方掠过，只被腿遮挡）。也就是说，我们 17×11 网格的**后半部分在真机上本来就是低置信甚至不可观测区域**，而现在的训练把它们当成和正前方一样准的测量。

再考虑点云密度与刷新率：

- 21600 有效点/s ÷ 11 Hz ≈ **约 2000 点/圈**，分布在 360°×90° 的球带里；落到"机体系 1.6×1.0 m 近场地面"上的点每圈只有几十到一两百个，**单圈必然有空洞**；
- L1 是**非重复扫描**（<https://www.unitree.com/cn/mobile/LiDAR>、拆机文 <https://developer.horizon.auto/blog/13003>），相邻圈的采样点不重合 → 单帧稀疏、需要多帧累积；
- 水平 11 Hz → 一圈约 90 ms；真机高程图一般 5–10 Hz 更新，因此**高度扫描的等效延迟约 100–200 ms**，而我们训练里是"当步即得"。这条延迟和空洞一样，是 §5 要随机化的对象。

---

## 2. 开源四足 RL 里 height scan 到底怎么做的

### 2.1 legged_gym / rsl_rl（ANYmal / Go1/Go2 常用基线）

- 实现在 _get_heights()：把机体系网格点按 **yaw-only** 旋转 + base 位置偏移 → 除以地形 horizontal_scale → 直接查**地形高度场张量** height_samples[px, py]（并做 2×2 的 min 采样），**根本不做射线投射**。因此不存在"打到自己"、不存在 FOV、不存在 miss。
  源码：<https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py>
- 观测构造：heights = clip(root_z − 0.5 − measured_heights, −1, 1) * obs_scales.height_measurements，默认 obs_scales.height_measurements = 5.0，且 mesh_type == 'plane' 时高度直接返回全 0。
- 网格默认 measured_points_x = [-0.8 … 0.8]（17 个）、measured_points_y = [-0.5 … 0.5]（11 个），即 **1.6×1.0 m、17×11 = 187 点**；measure_heights = True。配置：<https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py>
- 唯一的"退化"是**对整条 obs 加均匀噪声**：noise_scales.height_measurements = 0.1（实现是 obs += (2*rand−1)*noise_scale_vec，见 legged_robot.py）。
- rsl_rl 本身只吃 obs，不关心感知建模。

> **结论：legged_gym 的高度扫描是"完美解析高度场 + 加性噪声"，没有任何遮挡/盲区/miss 概念。** 我们的实现与它同源，只是把 base=0.5 改成 Go2 标定的 0.268、clip/scale 做了修正。

### 2.2 IsaacLab RayCaster（ANYmal / 各类 legged 任务）

- RayCasterCfg 的关键字段：mesh_prim_paths（要打的目标 mesh）、ray_alignment ∈ {base, yaw, world}、max_distance、drift_range、ray_cast_drift_range、pattern_cfg。
  源码：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py>
  文档：<https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/sensors/ray_caster.html>
- mesh_prim_paths 的说明是 **"Currently, only a single static mesh is supported"**：即高度扫描只打**静态地形 mesh**，机器人自身不在被投射目标里 → **不模拟机体/腿自遮挡**。
- drift_range / ray_cast_drift_range 模拟的是**位姿估计漂移**（float base 的 odometry drift），不是遮挡。
- 未命中（miss）语义：底层 raycast_mesh 对 miss 返回 float('inf')（源码注释明确写了 "The returned tensor contains float('inf') for missed hits"：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/warp/ops.py>），而 height_scan 观测函数直接 sensor_z − ray_hits_z − offset，**没有 miss 处理**：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/observations.py>。实践中地形是无限平面/大 heightfield，所以很少 miss；一旦为了模拟 FOV 而限制 max_distance，就会得到 −inf 甚至 NaN，需要自己加 miss_value。
- 新版 IsaacLab 文档提到：**PhysX/OV PhysX 后端**只加载配置的静态 mesh；**Newton 后端**改为查询 live scene BVH，"sees every collision shape in its environment, global terrain, and dynamic bodies"——这时机器人自身几何会参与射线求交（**这是最接近"自遮挡"的现成机制**，但它是"新物理后端的行为"，不是为遮挡建模设计的开关）。同一文档页。
- IsaacLab 也提供 RayCasterCamera/LidarPattern 等传感器，但 locomotion 的高度扫描主流仍是 mesh_prim_paths=[terrain] + ray_alignment="yaw"。

> **结论：IsaacLab 的 ray_caster 提供了"打什么 mesh / 打多远 / 位姿漂移"三个旋钮，但没有"未观测/被遮挡"的语义，也没有 FOV 盲区或自遮挡开关；miss 是 inf（未处理）。**

### 2.3 mjlab（MuJoCo-Warp 版 legged RL，与我们的技术栈最接近）

- 传感器：RayCastSensor + GridPatternCfg，支持 ray_alignment ∈ {base, yaw, world}、include_geom_groups、exclude_parent_body、max_distance、miss（距离 <0）语义。
  源码：<https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sensor/raycast_sensor.py>
- 通用 velocity 任务配置（velocity_env_cfg.py）里：

```python
terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
)
...
"height_scan": ObservationTermCfg(
    func=envs_mdp.height_scan,
    params={"sensor_name": "terrain_scan"},
    noise=Unoise(n_min=-0.1, n_max=0.1),   # actor
    scale=1 / terrain_scan.max_distance,
)
```

  源码：<https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/velocity_env_cfg.py>
- 注意 GridPatternCfg(size=1.6, resolution=0.1) 的生成方式（arange(-0.8, 0.8+0.05, 0.1) × arange(-0.5, 0.5+0.05, 0.1)）恰好是 **17×11 = 187 条射线**——我们仓库的 17×11 网格就是这套几何（只是把 x 范围前移成 [-0.4, 1.2]）。
- **miss 处理是显式的**：height_scan(..., miss_value=None) 默认 miss_value = sensor.cfg.max_distance，即未打中的射线填一个**常量（=5.0）**：<https://github.com/mujocolab/mjlab/blob/main/src/mjlab/envs/mdp/observations.py>。
- include_geom_groups=(0,) # Terrain only 明确**只打地形**，机器人自身不在求交集合里 → **不模拟自遮挡**；也没有 FOV（网格是"垂直向下平行射线"，不是真实雷达扇区）。
- actor 加 ±0.1 m 均匀高度噪声、critic 无噪（非对称 AC）。
- 当前 mjlab main 分支的 asset_zoo/robots 只有 unitree_go1、unitree_g1、i2rt_yam（无 Go2 资产）；velocity 任务的 config/ 也只有 g1、go1。所以"mjlab 的 Go2 ROUGH_TERRAINS_CFG"在 main 上**未找到**，但上面的 terrain_scan 实现是任务无关的，Go2 配置可复用同一套参数。

### 2.4 MuJoCo Playground（mujoco_playground）

- Go1 joystick 任务的 obs 是纯本体感知 + 指令：[local_linvel, gyro, gravity, joint_angles, joint_vel, last_act, command]，**没有 height scan**；地形默认 flat_terrain。
  源码：<https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py>
- 仓库 locomotion 目录里 Unitree 机型只有 go1（无 Go2）；barkour 等任务也是同一套 joystick 结构。即 **mujoco_playground 的 locomotion 没有地形高度感知，更没有遮挡建模**，不能直接作为"感知退化"参考。

### 2.5 横向对比

| 实现 | 高度来源 | 只打地形? | 模拟自遮挡? | 模拟 FOV 盲区? | 未命中处理 | 退化建模 |
|---|---|---|---|---|---|---|
| legged_gym | 直接查地形高度场张量（无 raycast） | 是（只有地形） | 否 | 否 | 不存在 miss | obs 加噪（height 0.1） |
| IsaacLab RayCaster | 静态地形 mesh 射线 | 是（单静态 mesh） | 否（PhysX）；Newton 后端会看到动态体（副作用） | 否 | miss = inf，未处理 | drift_range / ray_cast_drift_range + obs 噪声 |
| mjlab RayCastSensor | 地形 geom group 0 射线 | 是 | 否（exclude_parent_body 甚至主动排除本体） | 否 | miss = max_distance 常量 | actor 噪声 ±0.1 m |
| mujoco_playground | 无 height scan（平地） | — | — | — | — | 本体感知噪声 |
| **我们（现状）** | **解析高度场双线性采样** | 是 | **否** | **否** | **不存在 miss** | actor 高度噪声 0.02 m |

> **总结：开源四足 RL 的高度扫描普遍是"上帝视角完美高度场"的抽样（或"只打地形的理想射线"），没有一家把"机体/腿自遮挡、FOV 盲区、单帧空洞、感知延迟"建模进训练。** 这解释了为什么 sim2real 时"真机高程图有洞/有盲区"会成为策略没见过的新输入分布。

---

## 3. 高程图（elevation mapping）的遮挡处理

真机上的 height scan 通常不是"直接取雷达点"，而是：**点云/深度图 → 位姿估计 → 高程图（含 visibility/validity）→ 在机体系采样网格**。这一层已经有成熟的处理遮挡与空值的工程实践。

### 3.1 elevation_mapping_cupy（ETH RSL，ANYmal 系事实标准）

文档：<https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/introduction.html>；参数：<https://leggedrobotics.github.io/elevation_mapping_cupy/usage/parameters.html>；插件：<https://leggedrobotics.github.io/elevation_mapping_cupy/usage/plugins.html>；论文：<https://arxiv.org/abs/2204.12876>。

关键机制（均可在参数页/插件页核对）：

1. **地图是 robot-centric**：base_frame: 'base_link'（"This frame will be a center of the map"），corrected_map_frame: 'odom'。地图跟着机器人走，而不是固定的世界系。
2. **可见性清理（遮挡处理）**：enable_visibility_cleanup: true，配合 max_ray_length: 10.0（射线追踪最大长度）、cleanup_step: 0.1（"substitute this value from validity layer at visibility cleanup"）、cleanup_cos_thresh: 0.1。文档原话："**Visibility Cleanup and Artifact Removal**: Raycasting methods and an **exclusion zone** feature are designed to remove virtual artifacts and correctly interpret overhanging obstacles."——即用射线把"被前方障碍挡住的格子"标为不可见/清空，并排除机器人本体附近的区域。
3. **近距点直接丢弃**：min_valid_distance: 0.1（"points with shorter distance will be filtered out"）——所以即使雷达能测 0.05 m，进图也有 0.1 m 的内截断。
4. **高度范围门限**：max_height_range: 10.5（高于传感器的点被滤掉以屏蔽天花板）、ramped_height_range_a/b/c（随距离线性放宽的高度门限，模拟掠射角）。
5. **不确定度/时间衰减**：每个格点有 variance 层，time_variance: 0.0001（每次 update_variance 时加上）、max_variance: 100.0、initial_variance: 1000.0、initialized_variance: 10.0。语义是：**越久没被观测的格子方差越大**，下游可以据此降低信任——这就是"未观测格点"的显式表达（而不是偷偷填一个高度）。
6. **未观测格点的填充**：
   - min_filter 插件：fill_nan: True 时 "fills in minimum value around the invalid cell"（dilation_size=5, iteration_n=5）——用邻域最小值填，**偏保守（不会把坑填平）**；
   - inpaint 插件：基于图像修复（OpenCV inpaint）补大洞；
   - smooth、despiked（去掉孤立正向尖刺，保留沟槽）、near_base_filtered（配置示例中 radius_m: 4.0 / max_allowed_height_m: 0.5，用于处理靠近基座的点，防止本体点云污染）、robot_centric_elevation 插件（输出 robot-centric 层）。
7. **漂移补偿**：enable_drift_compensation: true、max_drift: 0.1、drift_compensation_alpha: 0.1——处理位姿估计漂移造成的图错位（对 sim2real 同样重要，因为我们的仿真没有位姿误差）。

### 3.2 grid_map 与 ANYmal 一系

- grid_map（ANYbotics）：多层 2.5D 栅格库，用**环形缓冲区**实现地图随机器人移动的非破坏性平移，本身不做遮挡推断，但为 elevation / variance / traversability 等层提供容器：<https://github.com/anybotics/grid_map>。它明确说是为 **Robot-Centric Elevation Mapping** 服务（<https://github.com/anybotics/elevation_mapping>）。
- 经典管线（ANYmal）：概率高程图 + 方差；后来 elevation_mapping_cupy 用 GPU 重写并增加 visibility cleanup / inpainting（论文 <https://arxiv.org/abs/2204.12876>：明确提到 "post processing steps ... such as smoothing, **inpainting** or plane segmentation"）。
- 面向"机器人中心高程图天生不完整"的**学习式补全**：2026 年 MDPI 论文 *Robot-Centric Elevation Map Completion with Sensor Geometry-Aware Augmentation and Uncertainty Estimation*：<https://www.mdpi.com/2076-3417/16/16/8262>。它把缺失来源列得很全（"limited sensor fields of view, occlusions, grazing-angle measurements, reflective or vegetated surfaces, **self-occlusion by the robot body**, and reduced density at longer range"），并用 **ray-cone augmentation**（训练时按传感器原点删掉角扇区）模拟缺失视角，报告"missing sensor sectors 上补全误差降低 8.3–9.7%"（原文另称洞区补全比最强经典插值好 45%，评测口径见原文）。**"结构化扇区遮挡"比随机像素 mask 更贴近真实缺失分布**——这条对我们直接可用。

### 3.3 "机体系高度扫描" vs "世界系高程图"

| | 世界系/odom 系高程图 | 机体系高度扫描（我们训练用的形式） |
|---|---|---|
| 表达 | 全局栅格，随时间累积、有记忆 | 每步以机器人当前位置重采样 |
| 遮挡/空洞 | 由 visibility + variance + inpainting 显式表达，**有 validity 层** | 训练侧通常**直接消失**（没有 miss 概念） |
| 位姿误差 | 会造成图错位（需要 drift compensation） | 训练侧无位姿误差 |
| 延迟 | 多帧融合 → 100–200 ms 级 | 当步即得 |
| 我们的仿真 | 不存在（解析高度场） | 每步"完美、无限分辨率、全可见" |

> 核心 gap：**真机喂给策略的是"世界/odom 系高程图在机体系的重采样结果"，天然带 validity、延迟、空洞和遮挡；我们的仿真喂的是解析高度场的理想采样。** 二者不是同一个信号。

---

## 4. 感知退化/遮挡下的鲁棒训练：已有工作与结论

| 工作 | 退化建模方式 | 结论/效果 | 链接 |
|---|---|---|---|
| Miki et al., Science Robotics 2022（ANYmal 野外） | 训练时给"不完整、有偏、带噪"的外感知；attention-based recurrent **belief encoder** 融合本体+外感知 | 遮挡/高反/雪地时"seamlessly fall back to robust proprioceptive locomotion"；作者明确指出**遮挡时高程图信息不足，策略会假设连续地面**（悬崖前会踏空），建议未来显式建模不确定度 | <https://arxiv.org/abs/2201.08117>，<https://perceptive-loco.github.io/> |
| DreamWaQ, ICRA 2023 | 只用本体感知 + **implicit terrain imagination**（用特权信息训练出的隐变量），AdaBoot 自适应 bootstrapping | 盲走也能过复杂地形；是"感知缺失时回退"的强 baseline | <https://arxiv.org/abs/2301.10602> |
| DreamWaQ++（Obstacle-Aware, 2024/25） | 多模态（本体+点云）；外感知编码器里加 **confidence filter / masking layer**，按统计特性把不可靠点特征置零；实验中**相机掉落/完全脱离**仍能走 | 证明"显式 mask 不可靠感知 + 保留盲走 backbone"有效 | <https://arxiv.org/abs/2409.19709> |
| RENet, 2025（视觉崩溃下的容错） | 双估计器（视觉 VP / 本体 OP）+ 逐环境 mask 选择 + 自编码器异常检测 + 低通切换；**raycaster randomization**：深度截断 2 m、1% 距离噪声 + 高斯噪声 σ²=0.01、±1 cm 位置 / ±5° 角度安装误差、边缘裁剪 | 视觉退化/未知环境时自动切到 OP，保持稳定；明确"vision blocked by sand/smog/occlusion → 依赖接触"的动机 | <https://arxiv.org/abs/2509.09283> |
| Beyond Egocentric Limits, 2025 | 多视角深度 + **stochastic remote-camera dropouts** + 3D 位置扰动 | 外视角部分/完全不可用时仍稳定；证明"训练时随机 dropout 视角"有效 | <https://arxiv.org/abs/2511.22744> |
| DPL, RA-L 2026 | **self-occlusion-aware ray casting** + noise-aware modeling 合成真实深度；blinded backbone | 地形重建误差降低 >30% | <https://arxiv.org/abs/2510.07152> |
| MDPI 2026 高程图补全 | **ray-cone augmentation**（按传感器原点删除角扇区）+ masked NLL 不确定度 | 结构化缺失下补全误差 −8.3~9.7%；比随机 mask 更贴近真实 | <https://www.mdpi.com/2076-3417/16/16/8262> |
| DAWN（2026，parkour） | 深度去噪世界模型，针对噪声鲁棒 | 噪声深度下 parkour 更稳 | <https://arxiv.org/abs/2609.29092>，<https://dawn-parkour.github.io/> |

归纳：

1. **"显式标出不可靠/缺失的感知"比"假装感知完好"更有效**：mask、confidence、validity、不确定度是反复出现的要素（DreamWaQ++、MDPI、elevation_mapping 的 variance 层）。
2. **训练时主动制造感知退化是共识**：加噪（RENet、legged_gym、mjlab）、几何遮挡（DPL、ray-cone）、视角 dropout（Beyond Egocentric）、整传感器失效（DreamWaQ++ 相机掉落、RENet 切换）。
3. **回退机制必须存在**：要么架构上（recurrent belief / blind backbone / 双估计器），要么在训练分布上（感知缺失时 critic 仍给正确梯度，actor 学会用本体）。
4. **结构化退化 > 随机退化**：ray-cone（扇区）比随机像素 mask 好（MDPI）；我们的"FOV+自遮挡"恰好就是结构化退化。
5. **未找到**：在 legged RL 训练中把"未观测高度格点统一填 0 + 全尺寸 mask 通道"作为标准做法的公开工作（有 mask/confidence 用于特征层，但没有明确的高度图 0+mask 方案）。这条需要我们自己定，并在 §5 给出可验证的设计。

---

## 5. 给我们这套训练的落地建议

前提：现有实现见 [envs/go2_walk.py](mjx-go1-getup/envs/go2_walk.py)（height_scan 配置在 360 行附近，height_scan_features() 在 742 行附近；actor 噪声 0.02 m、critic clean）。改动只影响 **actor 的 height scan 分支**，critic 保持 clean 特权信息（现有非对称 AC 不动），因此部署接口不变。

### 5.1 未观测格点的表达：推荐"0 填充 + 扇区可见性特征"

三个候选：

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| A（**推荐**） | 未观测格特征填 **0**（= 与标称站高相同，中性"平地"先验），另加 **8–12 维扇区可见性/置信特征**（每个扇区：可见比例、平均观测年龄、到最近可见格距离） | obs 维度只 +8~12；0 是中性值，与 clip/scale 后的特征空间自洽；置信特征让策略区分"真平地"与"不知道" | 单一 0 会丢空间细节 |
| B | 全尺寸 **187 维 mask 通道**（0/1），未观测格高度填 0 或上一次观测值 | 信息最全、实现直白 | obs 从 243 → 430（+77%），PPO 样本效率与推理成本上升；需要更多训练 |
| C | 未观测格**保持上一次观测**（last-known），无 mask | 接近真机 elevation map 的"记忆" | 训练侧要维护 per-env 历史 buffer（JAX 里要进 state，复杂度高）；策略无法知道这是旧值 |

建议：**先做 A**（低成本、可解释）；如果 A 的消融显示策略仍过度信任高度通道，再升级到 B。真机部署侧对应 elevation_mapping_cupy 的 validity/variance 层，语义一致（"不可见/高方差"→ mask=0 + 高度取 0）。

### 5.2 遮挡建模三件套（几何 + FOV + 随机）

**(a) FOV / 近场盲区（静态，可预计算）**

- 雷达原点（机体系）：lidar_xyz = (0.29, 0.0, lidar_z)，lidar_z 先用 0.27 m（URDF 口径），实机标定后更新（顶部安装则约 0.35–0.40 m）。
- 垂直可见条件：θ = atan2(lidar_z − h_ground, d) ≤ θ_down_max；d = hypot(x−0.29, y)。θ_down_max 未核实 → **参数化，先取 45°**（若 FOV 关于水平面对称则对应 ±45°），并用 30°/60° 做敏感性检查（对应盲区半径 0.47/0.27/0.16 m，见 §1.3）。
- 近距截断：d < 0.10 m 视为未观测（对齐 elevation_mapping_cupy 的 min_valid_distance；L1 本体 0.05 m 下限之上再留余量）。
- 因为机体系 yaw 对齐 + 机体几何固定，**这套 mask 与地形无关，可在 init 时算一次**（187 个 bool），运行时零成本。只有腿的摆动会改变遮挡，可先忽略（或加一条"腿部动态遮挡"近似：当某条腿的足端/小腿投影到某格点的射线上时遮挡，见 (c) 随机化里的腿遮挡 mask）。

**(b) 机体/腿自遮挡（几何近似）**

- **躯干**：URDF base collision box（0.3762 × 0.0935 × 0.114 m，中心在 base）。对每个格点，从雷达原点到该点地面做线段，与"base 站高 + 躯干盒"求交；相交则未观测。按 §1.3，在 θ_down=45°、lidar_z≈0.27–0.32 时，仅躯干+FOV 就有 **11%–22%** 的格点不可见；θ_down=30° 时 **37%–56%**。
- **腿**：4 条大腿+小腿可简化为胶囊（半径 3–4 cm，从 hip 到 knee、knee 到 foot 的 site_xpos）；射线与胶囊相交则遮挡。这一项对"脚下/侧前方"格点影响大，但它随步态变化 → 建议放进**运行时随机化**而不是每步精确算：每个 env 每步以一定概率把"当前腿投影覆盖的格点"置为未观测（近似用足端 xy 周围 0.15–0.25 m 半径的圆盘）。
- **成本**：187 格 × (1 盒 + 8 胶囊) 的线段求交在 JAX 里是几百次向量化运算/env/step，4096 envs 下约 10⁶ 级操作，和物理步比可忽略；但更推荐"静态躯干部分预计算 + 腿部圆盘随机化"，实现最简单。

**(c) 随机丢点 / 结构化遮挡（训练随机化）**

- **逐格随机丢点**：每个已观测格以概率 p_cell 变为未观测，独立同分布；建议 p_cell 课程 0 → 0.15。
- **结构化扇区遮挡**：以雷达原点为中心，随机删掉 1–2 个方位扇区（宽 45°–90°），或随机删掉 1–3 个矩形块（0.2–0.4 m 边长）；概率 p_sector 0 → 0.3。依据：MDPI ray-cone 证明结构化缺失比随机 mask 更真实、更有效（<https://www.mdpi.com/2076-3417/16/16/8262>）。
- **时间持续性**：遮挡状态用 AR(1) 保持（persistence 0.9–0.95），避免"每帧独立随机"被策略在时间上平均掉。真机空洞是持续 100–200 ms 的。
- **整帧失明（模拟 LiDAR 失效/极暗/强反光）**：每个 episode 以 p_blackout = 0.1–0.2 把整条 height scan 置为未观测（填 0 + mask 全 0）。依据：DreamWaQ++ 相机掉落实验、RENet 的 OP 回退、Miki 的"退回本体感知"（见 §4）。
- **高度噪声加大**：现有 noise=0.02 m 偏小；legged_gym 用 0.1、mjlab actor 用 ±0.1。建议 **可见格 U(−0.05, +0.05) m** + **5–10% 的格点加 ±0.15–0.2 m 离群噪声**（模拟玻璃/雪/植被/多路径），再加**全局高度偏置 U(−0.05, +0.05) m**（模拟高程图/状态估计的常值偏移，对应 IsaacLab 的 drift_range 与 elevation_mapping 的 drift compensation）。
- **感知延迟**：把 height scan 滞后 **0–4 个控制步**（50 Hz 下 0–80 ms；真机高程图 5–10 Hz + L1 11 Hz，等效 100–200 ms，建议课程末段覆盖到 0–4 步，必要时到 6 步）。实现上在 env state 里存一个 scan 历史队列（JAX 里 lax.scan 或 ring buffer）。
- **雷达安装误差**（可选）：雷达原点位置 ±1–2 cm、俯仰 ±2–5° 的随机化（RENet 用 ±1 cm/±5°，见 <https://arxiv.org/abs/2509.09283>）。

### 5.3 推荐参数表（可直接抄进 config）

| 参数 | 建议值 | 依据 |
|---|---|---|
| lidar_xyz（机体系） | (0.29, 0.0, 0.27) m，实机标定后更新；顶部安装假设用 0.35–0.40 | unitree_ros URDF + 本仓库站高 0.32 m；见 §1.2 |
| theta_down_max | 45°（默认），敏感性 30°/60° | L1 垂直 FOV 90°，上下界未核实；45° 是"关于水平面对称"的自然假设 |
| 近距截断 | 0.10 m | elevation_mapping_cupy min_valid_distance |
| 网格 | 保持 17×11、x∈[-0.4,1.2]、y∈[-0.5,0.5]、0.1 m | 与 mjlab GridPatternCfg(1.6,1.0,0.1) 同构；前移 x 是为台阶预判 |
| 未观测填充 | 0（中性） | 与 clip/scale 后空间自洽；"看不见≈假设平地"是 Miki 观察到的策略默认行为，但用 mask 提示不确定 |
| mask/置信特征 | 8–12 维（8 个方位扇区 × {可见比例, 最近观测年龄} 或 4 大区域 × 3 统计量） | 控制 obs 增量 <5%；部署侧可直接由 validity 层算 |
| p_cell（逐格丢点） | 课程 0 → 0.10–0.15 | 量级与 MDPI 缺失比例、legged_gym/mjlab 噪声强度一致 |
| p_sector（扇区遮挡） | 课程 0 → 0.15–0.30；扇区宽 45°–90°，1–2 个 | MDPI ray-cone（8.3–9.7% 提升）；RENet 边缘/视角随机化 |
| p_block（矩形块） | 0.2；块 0.2–0.4 m，1–3 个 | 模拟腿部/障碍物造成的局部空洞 |
| 遮挡持续性 | AR(1) 0.90–0.95 | 真机空洞持续 100–200 ms（L1 11 Hz / 高程图 5–10 Hz） |
| 高度噪声（可见格） | U(−0.05, +0.05) m | legged_gym 0.1 / mjlab ±0.1 的保守版；比现状 0.02 更真实 |
| 离群噪声比例 | 5–10% 格点 ±0.15–0.2 m | 模拟反光/雪/植被（Miki 2022 列出的失效源） |
| 全局偏置 | U(−0.05, +0.05) m（可慢变） | IsaacLab drift_range / elevation_mapping drift compensation |
| 感知延迟 | 0–4 步（可到 6） | L1 11 Hz + 高程图 5–10 Hz ≈ 100–200 ms |
| 整帧失明 | p_blackout 0.10–0.20 / episode | DreamWaQ++ 相机掉落、RENet OP 回退 |
| critic | 保持 clean 特权高度场（+ 可选全局偏置） | 现有非对称 AC；值函数不应被感知退化污染 |

### 5.4 课程安排（建议按训练进度百分比）

| 阶段 | 训练进度 | 遮挡设置 | 目的 |
|---|---|---|---|
| A | 0–30% | FOV + 躯干自遮挡 mask（固定）；p_cell=0, p_sector=0, p_blackout=0；噪声 0.02 | 先学会用"可见部分"的高度信息；避免一开始就瞎 |
| B | 30–55% | p_cell 0→0.10 线性；p_sector 0→0.20；高度噪声 0.02→0.05 | 逐步适应空洞与块状缺失 |
| C | 55–80% | p_cell=0.10–0.15；p_sector=0.25；p_block=0.2；p_blackout 0→0.15；延迟 0→2 步 | 加入结构化/整帧退化与延迟 |
| D | 80–100% | 全强度 + 延迟 0–4 步 + 离群噪声 10% + 雷达安装扰动；每 5M 步固定最难度 fine-tune | 收敛到"最坏情况也不崩" |

与现有地形课程（terrain_levels）联动：地形等级越高，遮挡强度上限越高（台阶/楼梯本身就制造视线遮挡）。依据：RENet 的两级课程、DreamWaQ 的 AdaBoot 都体现"先学会再退化"的顺序。

### 5.5 验收与消融（必须做，否则无法判断是否学到东西）

在 eval 里对同一策略跑 4 种 height scan 输入，报同样的指标（速度跟踪误差、摔倒率、通过率）：

1. clean（解析高度场）——上界；
2. 遮挡训练版本（FOV+自遮挡+随机）——目标；
3. **全 0 / 全 mask**——下界，近似"盲走"。

判据建议：

- 若 (3) 与 (2) 差距很小（<5%），说明策略已退化成盲走，高度通道没被真正利用 → 需要降低遮挡强度或加"必须用预判"的地形（如高台阶、宽沟）。
- 若 (2) 明显差于 (1)，说明退化建模过强或训练不足。
- 目标区间：(1) − (2) < 10–15%，且 (2) − (3) > 20–30%（即"有感知明显强于盲走，但感知退化时也不崩"）。这组判据来自 Miki/RENet 的"fallback 而非失效"的定性结论，具体阈值需我们自己的基线标定。

### 5.6 工程实现建议（JAX/warp 成本）

- **静态 mask 预计算**：FOV+躯干自遮挡只依赖机体系几何 → 在 env __init__ 里算一次 mask_static[187]（bool），并算好"每格点的方位扇区 id"。
- **每步只做随机化**：unobserved = ~mask_static | bernoulli(p_cell) | sector_mask | block_mask；feature = where(unobserved, 0, clip(...))；mask 特征由 unobserved 聚合到 8 扇区。这些都是 elementwise，成本可忽略。
- **可选的高保真模式**：如果以后想模拟"台阶边缘遮挡地面"（terrain self-occlusion），再实现对高度场的 ray-marching（每射线 8–16 步，步长 0.05–0.08 m）。这会把 terrain_height 调用量从 187 次/step 提到 ~1500–3000 次/step，**建议先测吞吐**；若太慢，可每 2–4 步更新一次 scan，或按 env 分片错峰更新。
- **不要用 IsaacLab 的 inf miss 语义**：若参考 IsaacLab 实现，必须自己加 miss_value（mjlab 的做法是填 max_distance，我们建议填 0 + mask，因为我们的特征语义是"相对高度差"而不是"到传感器的距离"）。
- 文档/日志：把"平均不可观测比例""整帧失明比例""延迟步数"写进训练日志，便于和真机统计对齐。

### 5.7 待实机标定/待核实项（影响参数取值）

1. L1 垂直 FOV 的上下界（本调研未核实，决定 theta_down_max 与 r_min）。
2. 雷达真实安装高度/倾角（URDF 0.28945/−0.046825/rpy 2.8782 需实机核对；决定 lidar_z 与自遮挡强度）。
3. 实机单帧点云在 187 格上的**可见率/空洞率**（建议用 ros2 topic echo /utlidar/cloud_deskewed 离线统计：每格在 1 s 内被点覆盖的比例）。
4. 真机高程图的实际更新率与延迟（决定延迟随机化上限）。
5. Go2 是否真的用 L1 出高程图（还是仅用于避障/建图）——若部署侧不用高程图而用别的感知，训练侧应对齐那个管线的统计特性。

---

## 6. 参考链接（按主题）

**Go2 / L1 / Mid-360**

- Go2 官方产品页（L1 360°×90°、0.05 m、Go2 标配）：<https://www.unitree-robot.com/go2/>
- Unitree 4D LiDAR L1 官方参数页：<https://www.unitree.com/cn/mobile/LiDAR>
- 百度百科 L1（引用官网参数）：<https://baike.baidu.com/item/Unitree%204D%20LiDAR%20L1/62957619>
- L1 拆机/原理分析（垂直 90°、双电机、非重复扫描、中心密度更高）：<https://developer.horizon.auto/blog/13003>，转载：<https://www.gansuan.com/tech/sharing/2776029949176213504.html>
- L1 SDK（坐标系定义：原点在底部安装面中心、+Z 向上、+X 背离线缆）：<https://github.com/unitreerobotics/unilidar_sdk>
- Go2 URDF（radar_joint xyz/rpy）：<https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf>
- 社区 ROS2 驱动 URDF（同组数值）：<https://github.com/abizovnuralem/go2_ros2_sdk/blob/master/go2_robot_sdk/urdf/go2.urdf>
- Go2 驱动 TF 链（radar → utlidar_lidar）：<https://ztl3106742440-hub.github.io/go2-tutorial/02-packages/06-driver/>
- Livox Mid-360 官方页（360°×59°、0.1 m、40 m@10%）：<https://www.livoxtech.com/mid-360>
- Livox Mid-360 用户手册 PDF（本次无法解析正文）：<https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Mid360/Livox_Mid-360_User_Manual_EN.pdf>

**开源四足 RL 的 height scan**

- legged_gym _get_heights / obs 构造：<https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py>
- legged_gym 配置（17×11、measure_heights、noise 0.1、obs_scale 5）：<https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py>
- IsaacLab RayCaster 文档：<https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/concepts/sensors/ray_caster.html>
- IsaacLab RayCasterCfg（mesh_prim_paths / ray_alignment / drift）：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/sensors/ray_caster/ray_caster_cfg.py>
- IsaacLab height_scan 观测：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/observations.py>
- IsaacLab raycast_mesh（miss = inf）：<https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/utils/warp/ops.py>
- mjlab RayCastSensor / GridPatternCfg：<https://github.com/mujocolab/mjlab/blob/main/src/mjlab/sensor/raycast_sensor.py>
- mjlab velocity 任务 terrain_scan 配置：<https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/velocity_env_cfg.py>
- mjlab height_scan 的 miss_value 语义：<https://github.com/mujocolab/mjlab/blob/main/src/mjlab/envs/mdp/observations.py>
- mujoco_playground Go1 joystick（无 height scan）：<https://github.com/google-deepmind/mujoco_playground/blob/main/mujoco_playground/_src/locomotion/go1/joystick.py>

**高程图 / 遮挡处理**

- elevation_mapping_cupy 简介（visibility cleanup / exclusion zone / drift）：<https://leggedrobotics.github.io/elevation_mapping_cupy/getting_started/introduction.html>
- elevation_mapping_cupy 参数（min_valid_distance、time_variance、cleanup_step、max_ray_length 等）：<https://leggedrobotics.github.io/elevation_mapping_cupy/usage/parameters.html>
- elevation_mapping_cupy 插件（min_filter / inpainting / smooth / robot-centric）：<https://leggedrobotics.github.io/elevation_mapping_cupy/usage/plugins.html>
- elevation_mapping_cupy 论文（GPU、inpainting、DARPA SubT）：<https://arxiv.org/abs/2204.12876>
- grid_map（多层 2.5D、环形缓冲、robot-centric）：<https://github.com/anybotics/grid_map>
- ANYbotics elevation_mapping（经典 robot-centric）：<https://github.com/anybotics/elevation_mapping>
- Robot-Centric Elevation Map Completion（ray-cone 遮挡增强、不确定度、补全误差 −8.3~9.7%）：<https://www.mdpi.com/2076-3417/16/16/8262>

**鲁棒感知/遮挡训练**

- Miki et al. 2022（belief encoder、遮挡下回退本体感知）：<https://arxiv.org/abs/2201.08117>，项目页：<https://perceptive-loco.github.io/>
- DreamWaQ（盲走 + 隐式地形想象）：<https://arxiv.org/abs/2301.10602>
- DreamWaQ++（confidence mask、相机掉落仍可走）：<https://arxiv.org/abs/2409.19709>
- RENet（视觉崩溃下的双估计器、raycaster randomization）：<https://arxiv.org/abs/2509.09283>
- Beyond Egocentric Limits（相机 dropout、多视角）：<https://arxiv.org/abs/2511.22744>
- DPL（self-occlusion-aware ray casting、深度合成）：<https://arxiv.org/abs/2510.07152>
- DAWN（噪声鲁棒 parkour）：<https://arxiv.org/abs/2609.29092>，<https://dawn-parkour.github.io/>

---

## 附录 A：§1.3 几何估算的方法与假设

估算脚本（纯 numpy，未进仓库、不影响训练）逻辑如下，**所有数字都是"假设下界"，不是实测**：

- 网格：x = linspace(-0.4, 1.2, 17)，y = linspace(-0.5, 0.5, 11)，共 187 格。
- 雷达原点：(0.28945, 0, h_lidar)，h_lidar ∈ {0.273, 0.32, 0.40} m（对应 URDF 口径 / base 高度 / 顶部安装）。
- 未观测判据：d < 0.10（近距截断）或 atan2(h_lidar, d) > θ_down（FOV）或"从雷达到格点地面点的线段与躯干盒相交"（自遮挡）。躯干盒：base 系 x∈[-0.1881,0.1881]、y∈[-0.04675,0.04675]、z∈[-0.057,0.057]，世界 z 加站高 base_h=0.32。
- 未含腿/头/地面台阶造成的遮挡，因此真实不可见比例**只会更高**。
- 复现：把上述逻辑写成一个 30 行脚本即可得到 §1.3 的两张表（本文档未附代码文件，避免改动仓库其他文件）。

## 附录 B：未找到 / 待核实清单（不编造）

| 项 | 状态 |
|---|---|
| L1 垂直 FOV 的角度上下界（如 ±45° 还是 0~-90°） | **未找到可引用的官方数值**。manualslib 上 L1 用户手册第 8 页标题为 "Installation; Effective Field Of View (FOV) Range"（<https://www.manualslib.com/manual/3274338/Unitree-4d-Lidar-L1.html?page=8>），但页面返回 403，无法读取正文；Unitree 下载中心页面（<https://www.unitree.com/cn/download/LiDAR/>）只列出文档名，未给出 PDF 直链。**结论：按 ±45° 假设并做 30/60° 敏感性分析，待实机核实。** |
| Go2 官方安装图纸/文字说明（雷达离地高度、倾角） | **未找到**官方文字；只有 URDF 数值（见 §1.2）。 |
| URDF rpy=(0, 2.8782, 0) 的物理含义（倒装？前倾 15°？） | **未找到**官方解释；数值本身可引用，解释需实机点云验证。 |
| Go2 实机点云在 17×11 网格上的可见率/空洞率统计 | **未找到**公开数据；建议自己采一次（§5.7 第 3 条）。 |
| Mid-360 垂直 FOV 上下界（-7°~+52°） | 官方页只给 360°×59°；具体上下界**未从官方文件核实**（PDF 不可解析）。 |
| "未观测高度格点统一填 0 + mask 通道"的 legged RL 标准做法 | **未找到**公开工作；只有特征层 mask/confidence（DreamWaQ++、RENet）和图像层 ray-cone mask（MDPI）。 |
| "Solving Occlusion in Terrain Mapping using Neural Networks"（NeurIPS 2021 workshop） | 找到标题索引，但原页面已 410（<https://dev.neurips.cc/virtual/2021/33699>），**未获取正文**，未采用其结论。 |
