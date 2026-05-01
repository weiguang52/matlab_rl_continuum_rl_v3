# Z 方向改为向下为正的修改说明

本版本将项目场景坐标统一为 `+Z` 向下，也就是 z 表示深度。主要变化：

1. `build_scene_model.m`：`scene.plateZ` 改为 `[0, 736.42, 1714.02, 2209.62, 2783.78]`。
2. `load_all_layer_data.m`：四层 `zRange` 改为正深度区间。
3. `continuum_forward_model.m`：直线段和常曲率段均改为沿 `+Z` 前进；`sampleCurvedSection` 内部同步改为 `transZ(+...)` 和 `rotY(+...)`。
4. `precompute_reachability.m` / `train_lower_sac.m` / `joint_evaluation.m`：`zProbe` 范围推算公式由负 z 版本改为正深度版本。
5. `render_scene_and_robot.m`：z 轴显示范围、视角、支撑柱上下边界更新为 +Z 向下。
6. `simulate_demo_rollout.m`：演示轨迹初始下探点从 `-100` 改为 `+100`。
7. 上层目标采样：原先负 z 约定下 `zRange(1)` 是下表面；现在 +Z 向下后下表面对应 `zRange(2)`。

注意：旧的 `reachability_lut.mat` 缓存不能复用，必须删除并重新运行预计算。
