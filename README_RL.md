# MATLAB 双层 RL 连续体机械臂规划系统（v2：多目标覆盖 + 可达性 LUT）

## v2 相对 v1 的主要变更

| 项目 | v1 | v2 |
|------|-----|-----|
| 上层任务 | 给 1 个目标，选 1 个入口（单步） | 给 N 个目标，序贯选多个入口，全覆盖（多步） |
| 优化目标 | 距离 + 障碍惩罚 | **优先：最少定位点；其次：最少下探孔** |
| 可达性判断 | 仅 2D 直线 + 障碍检测 | **真实 6-DoF 正运动学 → 离线预计算 LUT** |
| 目标 z 坐标 | 各层 zRange 中点 | **各层 zRange(2)**（+Z 向下后的下表面/较深边界） |
| 目标 XY | 连续随机 | 连续随机 + 拒绝采样（排除障碍内 / 不可达区） |

## 文件结构

| 文件 | 功能 |
|---|---|
| `main_rl_train.m` | 主入口：场景 → 可达性 LUT → 上层 DQN → 下层 SAC → 联合评估 |
| `precompute_reachability.m` | **NEW**：离线蒙特卡洛预计算每入口可达性网格 |
| `train_upper_dqn.m` | **重写**：多目标覆盖 MDP + 定位点/下探孔最小化 |
| `train_lower_sac.m` | 下层 SAC（与 v1 相同） |
| `joint_evaluation.m` | **重写**：序贯规划 → 下层 rollout → 统计定位点/下探孔/成功率 |

## 快速开始

```matlab
cd matlab_rl_continuum_rl
main_rl_train
```

首次运行会预计算可达性 LUT（约 10~25 分钟，单核），缓存到 `reachability_lut.mat`，
后续运行直接加载（秒级）。

---

## 核心设计：多目标覆盖 MDP

### 状态 (10 + nAnchors 维)

```
[ 层 one-hot (4),
  未覆盖目标数比例 (1),
  未覆盖目标 XY 质心归一化 (2),
  未覆盖目标 XY 包围盒尺寸归一化 (2),
  已使用定位点 one-hot 并集 (nAnchors),
  已使用下探孔数归一化 (1) ]
```

**为什么这样编码？**

目标集是变长的，不能直接喂给全连接网络。使用
*质心 + 包围盒* 作为低维固定维度摘要，
配合"数量比例 + 已用资源"足够支撑贪心覆盖决策。

### 动作

当前层所有 RobotArm 类型入口的离散索引 `a ∈ {1, …, nEntry_L}`，
训练/推断时可选屏蔽已用入口（`opts.mask_used_holes = true`）。

### 转移

选定 `entry = entries(a)` 后：
1. 查 `reach.query(layerId, a, target_xy)` → 把 entry 可达域内的未覆盖目标标为已覆盖
2. 将 `entry.anchorPoint` 加入已用定位点集合
3. 将 `a` 加入已用下探孔集合

### 奖励（负代价设计，Double DQN 最小化）

```
r_step = R_NEW_ANCHOR  (-15)   若本步引入新定位点
       + R_NEW_HOLE    (-1.5)  每用一个新下探孔
       + R_NO_PROGRESS (-3)    若本步未覆盖任何新目标
       + R_PROGRESS_SCALE · n_newly   (+0.2 · n)  覆盖新目标的小奖

r_term = +R_COVER_BONUS (+20)   全覆盖
       OR  R_FAIL       (-25)   超时未全覆盖
```

**优先级实现**：
`|R_NEW_ANCHOR| : |R_NEW_HOLE| = 15 : 1.5 = 10`。
即使多用 10 个下探孔也比多用 1 个新定位点便宜，
保证策略首先压定位点数量，其次压下探孔数量。

---

## 可达性查找表（reach LUT）

### 为什么需要

v1 只用二维直线 + 障碍检测判断可达性，完全忽略运动学约束。
v2 必须真实判断"从某入口出发，6-DoF 关节空间内是否存在一个
无碰撞关节构型使末端到达目标 XY"。
这本质上是 IK 可达性问题，在线算代价太大（几十万次 FK）。

### 离线预计算做法

对每个入口（共约 350 个）：
1. 在合法关节范围内**蒙特卡洛采样 4000 组 q**
2. 调用 `continuum_forward_model` 计算末端位置
3. 过滤：tip 的 z 必须落在本层 zRange；`collision_check_centerline` 必须通过
4. 将 tip XY 打到 30mm 分辨率网格
5. 一次形态学膨胀（1 格）填补采样稀疏的洞
6. 缓存每入口的 `logical(H, W)` 可达网格到 `.mat`

### 查询

```matlab
ok = reach.query(layerId, entryIdx, targetXY);  % O(1) 查表
ok = reach.query(layerId, [],        targetXY); % 该层任一入口可达
```

### 参数调整

```matlab
reach_opts.cellSize     = 30;    % 网格精度（越小越精确，但计算量 O(n²)）
reach_opts.N_samples    = 4000;  % 采样数（越多越稠密但越慢）
reach_opts.dilateRadius = 1;     % 膨胀半径（小=保守，大=激进）
```

快速测试：`N_samples = 1500, cellSize = 50` 约 2 分钟完成；
生产环境：`N_samples = 8000, cellSize = 20` 约 1 小时。

---

## 课程学习

| 阶段 | Episode | 层 | 目标数 | 范围 |
|------|---------|-----|--------|------|
| 0 | 0 ~ 400 | LGP | 5~8 | XY∈[-400, 400]² |
| 1 | 400 ~ 1200 | 随机 | 5~15 | 全层 |
| 2 | ≥ 1200 | 随机 | 5~15 | 全层 |

---

## 主要超参数

```matlab
% 上层 DQN
opts_upper.num_episodes   = 2000
opts_upper.max_steps      = 15        % 单 ep 最多选 15 次入口
opts_upper.lr             = 3e-4
opts_upper.gamma          = 0.98
opts_upper.batch          = 64
opts_upper.hidden         = [256, 256]
opts_upper.n_targets_min  = 5
opts_upper.n_targets_max  = 15

% 奖励权重（体现优先级）
opts_upper.R_NEW_ANCHOR   = -15.0     % 新定位点（主惩罚）
opts_upper.R_NEW_HOLE     = -1.5      % 新下探孔
opts_upper.R_NO_PROGRESS  = -3.0
opts_upper.R_COVER_BONUS  = +20.0
opts_upper.R_FAIL         = -25.0

opts_upper.mask_used_holes = true     % 推断时屏蔽已用孔
opts_upper.require_reachable = true   % 只采样可解目标
```

---

## 训练输出

- `reachability_lut.mat`：可达性查找表（可复用）
- `upper_dqn_result.mat`：上层 DQN 权重、训练曲线、anchorTable
- `lower_sac_layerN.mat`：各层下层 SAC 权重
- `rl_rollout.gif`：联合评估中一条成功轨迹的 3D 动画

---

## 训练监控

训练过程每 50 ep 打印：
```
[Upper DQN] ep=850 phase=1 eps=0.47 meanR=-22.3 | near50 avgAnchors=2.32 avgHoles=3.86 cov=0.97
```
- `avgAnchors` 下降 → 定位点节省（主目标）
- `avgHoles` 下降 → 下探孔节省（次目标）
- `cov` → 1 → 全覆盖约束被满足

理想收敛轨迹：`cov` 先升到 1，然后 `avgAnchors` 和 `avgHoles` 持续下降。

---

## 联合评估指标

```
下层到达成功率       真实 SAC rollout 能否到达目标
上层规划覆盖率       上层序贯决策后的可达域是否覆盖全部目标
下层碰撞率           SAC 执行中发生碰撞的目标比例
平均定位点数 / ep    主优化指标
平均下探孔数 / ep    次优化指标
平均步数 / 目标      下层 SAC 收敛效率
```

---

## 依赖

- MATLAB R2022b+（Deep Learning Toolbox：`dlnetwork` / `dlfeval`）
- 不依赖 Reinforcement Learning Toolbox
- 不依赖 Image Processing Toolbox（膨胀手写实现）
