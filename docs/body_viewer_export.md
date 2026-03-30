# `body_viewer_export_v1` 契约草案

最后更新：`2026-03-18`

## 1. 目的

这份文档冻结 `dyadic_motion_model -> body_viewer` 的第一版对接契约。

目标只有一个：

- 让 `dyadic_motion_model` 未来的模型输出，能够通过一个明确、稳定、可版本化的导出 adapter，被下游共享仓库 `body_viewer` 直接消费。

这份契约刻意不反向定义训练主空间。

- 训练主空间可以是 `body 258`、开放 `face drive 33`、未来的 `joint 291`
- 但导出给 `body_viewer` 时，必须统一落到本契约定义的字段集合

## 2. 当前 `body_viewer` 的真实消费口径

`body_viewer` 当前直接消费的是一份与 `Seamless Interaction` 原始 `.npz` 兼容的字段化表示，而不是单个拼接向量。

它实际依赖的键分成 4 组：

1. body / hand：
   - `smplh:global_orient`
   - `smplh:body_pose`
   - `smplh:left_hand_pose`
   - `smplh:right_hand_pose`
   - `smplh:translation`
2. face drive：
   - `movement:FAUValue`
   - 如果存在 `movement_v4:expression`，可直接走 `Direct FLAME` 模式
3. head alignment：
   - `movement:alignment_head_rotation`
   - `movement:alignment_translation`
   - `movement:is_valid`
4. gaze：
   - 当前优先从 companion CSV 的 `gaze_x / gaze_y` 读取
   - 不依赖 `movement:gaze_encodings` 作为绝对 gaze 角度来源

因此，`body_viewer` **不直接消费**：

- `Imitator 137`
- `joint 395`
- `body_motion_258`
- `ARKit52`

这些空间如果要对接 `body_viewer`，都必须先经过导出 adapter。

## 3. 导出产物布局

`body_viewer_export_v1` 推荐产物布局如下：

```text
<sample_root>/
  <sample_id>.npz
  <sample_id>_metadata.json         # 推荐
  <sample_id>_blendshapes.csv       # 可选，供 gaze_x/gaze_y
```

其中：

- `.npz` 是强制主产物
- `_metadata.json` 是推荐 sidecar，用于记录来源、占位字段、模型版本
- `_blendshapes.csv` 是可选增强项；只有当你要给 `body_viewer` 提供绝对 gaze 角度时才需要

## 4. 必填 NPZ 字段

### 4.1 body / hand

| 字段 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `smplh:global_orient` | `[T, 3]` | `float32` | root global orientation，axis-angle，单位 `rad` |
| `smplh:body_pose` | `[T, 21, 3]` 或 `[T, 63]` | `float32` | 21 个 body joints 的 axis-angle，单位 `rad` |
| `smplh:left_hand_pose` | `[T, 15, 3]` 或 `[T, 45]` | `float32` | 左手 15 joints 的 axis-angle，单位 `rad` |
| `smplh:right_hand_pose` | `[T, 15, 3]` 或 `[T, 45]` | `float32` | 右手 15 joints 的 axis-angle，单位 `rad` |
| `smplh:translation` | `[T, 3]` | `float32` | root translation，单位与原 `Seamless` 兼容 |

### 4.2 face drive

| 字段 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `movement:FAUValue` | `[T, 24]` | `float32` | 与 `body_viewer` 当前 `FacePipeline` 兼容的 24 维 FAU/FACS 强度 |

说明：

- `body_viewer_export_v1` 的 face 主字段先冻结为 `movement:FAUValue`
- 这意味着工程主线的 face 输出必须能稳定 decode 到 24 维 FAU
- 不要求训练主空间本身就是 24 维，但导出时必须能落到这个字段

### 4.3 head alignment

| 字段 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `movement:alignment_head_rotation` | `[T, 3]` | `float32` | 头部对齐旋转，axis-angle，单位 `rad` |
| `movement:alignment_translation` | `[T, 2, 3]` | `float32` | 头部/对齐平移，保留 `Seamless` 兼容 shape |
| `movement:is_valid` | `[T]` 或 `[T, 1]` | `bool` / `float32` | face/head track 是否有效 |

说明：

- 即使 face 模型当前还不完整，这 3 个字段也必须始终存在
- `body_viewer` 的头姿链直接依赖它们

## 5. 推荐可选字段

### 5.1 direct FLAME

| 字段 | shape | dtype | 语义 |
| --- | --- | --- | --- |
| `movement_v4:expression` | `[T, 50]` | `float32` | 50 维 FLAME expression coefficients |

说明：

- 这是 `body_viewer` 的长期优选增强出口
- 一旦你有稳定的 face decoder，建议优先补这个字段
- 有了它，`body_viewer` 可以绕过 `FAU -> ARKit -> FLAME`，直接走 `Direct FLAME`

### 5.2 gaze companion CSV

推荐 CSV 列：

| 列名 | 语义 |
| --- | --- |
| `frame_idx` | 帧号 |
| `gaze_x` | 竖直 gaze 角度，单位 `deg` |
| `gaze_y` | 水平 gaze 角度，单位 `deg` |

说明：

- 当前 `body_viewer` 绝对 gaze 角度优先从 CSV 读取
- `movement:gaze_encodings` 不应作为 `body_viewer_export_v1` 的强制字段

## 6. 从当前模型输出到本契约的映射

### 6.1 `body_motion_258 -> body_viewer_export_v1`

当前已有清晰映射：

- `body_motion_258 [T, 258]`
- `43 x 6D rotation`
- `43 x axis-angle`
- scatter 回：
  - `smplh:body_pose`
  - `smplh:left_hand_pose`
  - `smplh:right_hand_pose`

当前工程阶段的额外限定必须明确写死：

- `body_motion_258` 只覆盖上半身与双手
- 下半身关节不在当前训练输出里
- 当前工程主线不学习 `smplh:global_orient`
- 当前工程主线不学习 `smplh:translation`

还需要补的导出字段：

- `smplh:global_orient`
- `smplh:translation`

在没有更强预测器前，允许由 adapter 提供：

- 固定直立根姿态模板
- 固定站位模板
- 或来自条件侧 / seed / GT sidecar

当前默认固定模板收敛为：

- `smplh:global_orient = [0.0, 0.0, pi]`
- `smplh:translation = [0.0, 0.0, 0.63]`

补充约束：

- 零值占位只允许用于 `stream_to_unreal.py` 级接口 smoke
- `view_motion.py` 级播放器验收不允许直接使用零根姿态/零根平移占位
- 当前播放器口径必须理解为“固定下半身 + 固定根姿态/站位 + 预测上半身/双手/face”

### 6.2 开放 `face drive 33 -> body_viewer_export_v1`

当前建议冻结的工程主线 face 输出为：

\[
face\_drive\_{33} = [fau_{24}, head\_rot_{3}, head\_trans_{6}]
\]

其中：

- `fau_24` -> `movement:FAUValue`
- `head_rot_3` -> `movement:alignment_head_rotation`
- `head_trans_6` -> reshape 成 `movement:alignment_translation [T, 2, 3]`

`movement:is_valid` 作为并行 mask 单独导出，不计入 `33-d` 主向量。

### 6.3 `joint 291 -> body_viewer_export_v1`

当前建议冻结的工程主线 joint 输出为：

\[
joint\_{291} = [face\_drive\_{33}; body\_motion\_{258}]
\]

这里必须同步明确：

- `joint_291` 只表示 `face + upper-body/hands`
- 不表示下半身
- 不表示 `smplh:global_orient`
- 不表示 `smplh:translation`

导出时必须先 split：

- `33 -> face fields`
- `258 -> smplh fields`

再叠加固定下半身模板与固定根姿态/站位模板，组装成 `body_viewer_export_v1.npz`

### 6.4 `Imitator 137 -> body_viewer_export_v1`

`Imitator 137` **不能直接导出** 到 `body_viewer_export_v1`。

如果以后要接它，必须先显式补一个 decoder，把它转换成以下至少一种开放空间：

1. `FAU24 + head_rot3 + head_trans6`
2. `FLAME50 + head_rot + head_trans`

在 decoder 明确存在之前，`Imitator 137` 不得再被视为当前主线对下游交付的 face 契约。

## 7. 兼容级别

### Level A：body-only smoke

允许：

- `movement:FAUValue = 0`
- `movement:alignment_head_rotation = 0`
- `movement:alignment_translation = 0`
- `movement:is_valid = 1`
- `smplh:global_orient / translation` 可临时占位

用途：

- 先验证 body 动作、骨架、网格、SMPL-H/SMPL-X 对接
- 不作为最终播放器口径

### Level B：collab-ready face+body

要求：

- 非占位 `movement:FAUValue`
- 非占位 head alignment
- `smplh:*` 与 `translation` 全量可播
- 下半身来自固定站立模板
- `smplh:global_orient` 来自固定直立模板或 sidecar
- `smplh:translation` 来自固定站位模板或 sidecar

用途：

- 与 `body_viewer` 下游协作时的默认交付级别

### Level C：direct-FLAME enhanced

在 Level B 基础上，再提供：

- `movement_v4:expression`
- 可选 gaze CSV

用途：

- 提升 face fidelity
- 减少 `FAU -> ARKit -> FLAME` 间接链带来的信息损耗

## 8. 版本规则

- 这份契约的版本号固定为 `body_viewer_export_v1`
- 后续如果字段名、shape、语义发生 break change，必须升到 `v2`
- 新增可选字段不算 break change

## 9. 当前冻结结论

当前项目阶段里，最稳妥的工程主线是：

1. body 继续用 `258`
2. face 工程主线改为开放 `face_drive_33`
3. 对下游统一导出 `body_viewer_export_v1`
4. `Imitator 137` 论文忠实 research 支线当前记为暂停，不再阻塞当前主线
