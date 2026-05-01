# MATLAB 双层 RL 连续体机械臂离线规划示例

本工程给出一套**可运行的 MATLAB 代码骨架**，用于：

1. 解析四层板间的 `SupportInfo` 与 `ExplorationInfo` 文本数据；
2. 构建五层板 / 四个板间空间 / 支撑柱 / 可选下探孔的三维场景；
3. 上层离散策略：选择 `AnchorPoint + 下探孔`；
4. 下层连续控制：在固定入口孔后，逐步控制两段连续体的弯曲与旋转，使末端到达目标点并避障；
5. 给出一个不依赖训练结果的**演示轨迹**，直接生成三维动画 `demo_rollout.gif`。

## 目录说明

- `main_demo.m`：主入口，构建场景并输出动画
- `run_upper_qlearning_demo.m`：上层离散 Q-learning 示例
- `run_lower_sac_setup.m`：下层 SAC 环境与 agent 创建脚本（需 Reinforcement Learning Toolbox）
- `load_all_layer_data.m`：读取四个 Support/Exploration 文件
- `build_scene_model.m`：构建四个板间空间的几何模型
- `choose_entry_rule_based.m`：基于几何难度的上层示例策略
- `simulate_demo_rollout.m`：固定入口后生成一个示例避障轨迹
- `render_scene_and_robot.m`：三维场景与机器人绘图
- `continuum_forward_model.m`：连续体机器臂常曲率前向形状模型（参考用户提供代码重构）
- `collision_check_centerline.m`：基线对安全柱碰撞判定
- `parse_support_info.m` / `parse_exploration_info.m`：文本解析
- `grid_label_to_xy.m` / `hole_label_to_xy.m`：坐标映射

## 数据文件放置

把下面 8 个文件与代码放在同一目录，或在 `main_demo.m` 中修改 `dataDir`：

- `CSP_SupportInfo(7).txt`
- `LGP_SupportInfo(7).txt`
- `SGP_SupportInfo(7).txt`
- `BP_SupportInfo(7).txt`
- `CSP_ExplorationInfo(7).txt`
- `LGP_ExplorationInfo(7).txt`
- `SGP_ExplorationInfo(7).txt`
- `BP_ExplorationInfo(7).txt`

## 快速开始

```matlab
cd matlab_rl_continuum_demo
main_demo
```

运行后会：

1. 解析数据
2. 构建四个板间空间的柱子/下探孔/可选入口
3. 随机或指定目标点
4. 选择一个入口
5. 生成一个示例避障轨迹
6. 输出 `demo_rollout.gif`

## RL 建议

### 上层（离散）

状态：
- 目标层 `layerId`
- 目标 `x, y`
- 目标附近局部障碍统计

动作：
- 当前层所有可选 `RobotArm` 入口 `(AnchorPoint, ExplorationNum)`

奖励：
- 到达可解入口：`+10`
- 更换定位点惩罚：`-w_anchor`
- 入口到目标平面距离惩罚：`-w_dist * d_xy`
- 若入口与目标之间需要绕柱超过 1 根：大惩罚

建议先用 Q-learning / Double DQN。

### 下层（连续）

观测：
- 当前末端位置误差 `e = p_target - p_tip`
- 当前两个弯曲角 / 两个旋转角 / 下探深度
- 最近障碍物的相对位置与距离
- 是否穿过目标板层窗口

动作（连续）：
- `ΔzProbe`
- `Δalpha1, Δtheta1`
- `Δgamma,  Δtheta2`
- 可选 `Δalpha3`

奖励：
- 靠近目标奖励：`k1 * (d_prev - d_now)`
- 成功到达奖励：`+R_goal`
- 碰撞：`-R_collision`
- 超限：`-R_limit`
- 大动作平滑惩罚：`-k2 * ||u||^2`

建议用 SAC / TD3。

## 说明

本示例中 `main_demo.m` 的动画轨迹为了可直接展示效果，采用了“目标吸引 + 单柱绕行偏置 + 碰撞回退”的启发式控制，便于直接看场景与运动效果。真正训练时请运行 `run_lower_sac_setup.m` 并在 MATLAB 内开启训练。
