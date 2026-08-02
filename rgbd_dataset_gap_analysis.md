# LIBERO-Plus RGB-D v1/v2 差异与验证报告

## 结论

v1 的 72 条 success-selected RGB-D 样例已通过工程验证（GO）。v2 已恢复 192 条官方参考的唯一映射，冻结 216 次尝试并生成 72 条 HDF5、24 个预览；完整验证和 6/6 确定性复跑均通过，结论为 **GO**。另生成了 6 条 action 指纹完全相同、逐 action index 对齐的复刻 / 官方对比视频。v1 文件未被修改。

因此，v2 证明固定归档连接、有限参考校准和 RGB-D 生成管线可闭环，但 72 条 success-selected 样例仍不等于复现完整官方训练分布。Depth 仍没有官方真值。

## v2 官方参考审计结果

| 项目 | 结果 | 结论 |
|---|---:|---|
| 固定归档缓存 | 7 个分卷文件，约 159 GiB | RLDS、Camera、Segmentation 均按固定 revision 和 LFS SHA-256 校验 |
| 官方参考 | 192 | 6 settings × 4 tasks × 8 条 |
| Setting 分布 | 每类 32 | Objects、Background、Light、Camera、Language、Noise 完整 |
| Calibration / holdout | 144 / 48 | 每个 `task × setting` 稳定 6/2 切分，组内 logical episode 无泄漏 |
| Action / state 最大误差 | `0 / 0` | 全部满足 `≤1e-6` |
| TFRecord CRC | 192/192 | 全部通过 |
| 官方帧方向 | 192/192 已旋转 180° 后与复刻 renderer 对齐 | 固定归档的 RGB/mask 相对本地 renderer 同时上下、左右反向，统一在解码入口旋转 180°；不是水平或竖直单轴镜像。raw record/frame hash 不变；修复后 8 个槽的最终候选改变并已重建 |
| 映射置信度 | 168 条 high、24 条 medium | 24 条 Language 参考因跨任务补足而降级；全部 revision、volume、member、record、raw/frame hash 仍可追踪 |
| 扫描范围 | Camera 256；Segmentation 1024 shards | 使用本地完整归档；没有建立 14,347 条持久索引 |
| Calibration probes / attempts | 576 / 216 | 每槽 8 个 probe，冻结最低 loss 的 3 次 retry |
| v2 HDF5 / 预览 | 72 / 24 | 4 suites × 6 settings × 3 slots，覆盖完整 |
| 实际 attempts | 73 | 72 成功、1 失败；`libero_10.objects.slot1.try1` 未完成任务，按冻结 manifest 的 retry 2 成功，失败证据保留 |
| Suite / setting 分布 | 每 suite 18；每 setting 12 | 与冻结协议一致 |
| 确定性重跑 | 6/6 | 每类 setting 一条内容 hash 一致 |
| 最终结论 | **GO** | 六类 calibration / holdout loss 均为有限数值 |

| Setting | 样例 | 官方参考（calibration/holdout） | Calibration loss | Holdout loss | 置信度 |
|---|---:|---:|---:|---:|---|
| Objects | 12 | 32（24/8） | 0.731885 | 0.734342 | high |
| Background | 12 | 32（24/8） | 12.9921 | 12.6489 | high |
| Light | 12 | 32（24/8） | 0.115675 | 0.253762 | high |
| Camera | 12 | 32（24/8） | 0.390672 | 0.470594 | high |
| Language | 12 | 32（24/8） | 97222.2 | 97222.2 | medium |
| Noise | 12 | 32（24/8） | 0.265249 | 0.317787 | high |

机器可读映射见 [`reference_mapping.jsonl`](data/libero_plus_rgbd_sample_v2/evidence/reference_mapping.jsonl)，参考内容见 [`official_references.hdf5`](data/libero_plus_rgbd_sample_v2/evidence/official_references.hdf5)，数值结果见 [`calibration.json`](data/libero_plus_rgbd_sample_v2/reports/calibration.json)，最终验收见 [`validation.json`](data/libero_plus_rgbd_sample_v2/reports/validation.json) 和 [`gap_report.md`](data/libero_plus_rgbd_sample_v2/reports/gap_report.md)。

## v2 与官方数据集的逐项差异

下表中的“官方数据”指固定 revision 的 RLDS、Camera 和 Segmentation 归档；“v2 复刻”指本次生成的 72 条 RGB-D 样例。这里的复刻是 replica 语义，不是对图像做 mirror。192 条参考只覆盖 4 个代表任务，不代表全部 14,347 条官方 episode。各类 loss 使用不同特征空间，只能在同一 setting、同一 metric version 内解释，不能横向比较大小，也没有“足够接近官方”的通过阈值。

| 差异 | v2 复刻当前能保证什么 | 与官方数据仍有何不同或缺少什么 | 当前结论 |
|---|---|---|---|
| Front RGB | 复刻与官方归档均保存完整的 `256×256` front RGB；官方 RGB/mask 在共享解码入口旋转 180° 后再进入 reference feature；6 条视频另行使用相同 action 指纹并逐 index 对齐 | 校准本身是集合级特征近邻，不是逐像素拟合；官方 texture、seed 和渲染参数未公开。raw/frame hash 始终取变换前的归档字节 | 可以比较有限参考上的特征分布和相同控制轨迹的画面，不能声称逐像素复刻 |
| Wrist RGB | 复刻与官方归档都包含完整 wrist RGB；复刻还保存逐帧 wrist 内外参和米制 depth；对比视频按与 front 相同的 action index 展示 wrist | 当前 calibration/loss 没有使用 wrist RGB；官方归档没有 wrist depth，也没有可直接核验的 wrist extrinsics | 可以做同步定性比较，但没有官方 wrist 数值 gap |
| 图像方向 | 视频左侧直接读取复刻 HDF5，右侧对官方归档 JPEG 旋转 180°；front/wrist 均使用相同的左上角显示原点，2×2 非对称像素回归测试同时约束两条轴 | 归档为何采用该存储方向没有随数据发布；只对解码结果做坐标归一化，原始记录和 hash 不改 | 两侧不再互为镜像；剩余画面差异来自场景随机化、相机和渲染参数，而非人为水平翻转 |
| Compounding Objects | 官方 Segmentation 参考包含实例 mask 和 object map；复刻固定 1/2/3 个额外对象槽，并按额外对象数、类别 Jaccard、mask 面积和质心选择候选 | 复刻使用本地 BDDL 候选及可成功 replay 的放置；没有恢复官方逐 episode 对象 ID、pose、placement seed 或失败放置 | Calibration / holdout loss 为 `0.731885 / 0.734342`；这是方向对齐后的 mask 特征级近邻，不是对象配置复原 |
| Background | 复刻用前景补集的 RGB histogram、亮度和多尺度 Sobel 特征，从 calibration pool 选择 3 个 medoid | 没有官方逐 episode texture/material 参数、资产 ID、UV、seed；复刻 medoid 也不是对应官方 episode 的原始背景 | Loss 为 `12.9921 / 12.6489`；已量化有限参考差异，但不能恢复官方纹理生成过程 |
| Light | 复刻用亮度、色度/CCT、阴影和高光比例选择 25/50/75 分位候选，并保存实际模型 XML | 官方 Light 归档提供 RGB 和 setting 路径，但不提供逐 episode 光源位置、颜色、强度或 seed | Loss 为 `0.115675 / 0.253762`；只覆盖画面统计，不等于光源参数一致 |
| Camera | 官方 Camera 参考提供 primary-camera extrinsics；读取时已把原始 MuJoCo camera axes 转为 robosuite/OpenCV 约定，pose 特征保留有符号旋转向量和位移向量；复刻保存逐帧 front/wrist 外参 | 官方没有对应 wrist extrinsics；复刻只能从 8 个冻结候选中逼近，不能复用未公开的逐 episode camera 参数。相同 action 的代表视频仍有 `4.24665° / 0.154755 m` front pose residual | 修复后 loss 为 `0.390672 / 0.470594`；旧值 `4.89106 / 4.94660` 来自坐标轴约定错误，已作废 |
| Language | 复刻 rewrite 通过实体、目标、动作和空间关系硬校验，并保存固定 Qwen revision、prompt 与 tokenizer hash；代表视频使用相同任务和相同 action | 官方公开记录在单任务内基本只有 canonical instruction；为构造不同目标，32 条 Language calibration 参考中仍有 24 条跨任务补足 | Loss 为 `97222.2 / 97222.2`、confidence 为 medium；这是最弱的一类证据，不代表恢复了官方 rewrite 原句或生成过程 |
| Sensor Noise | 复刻保存算法、severity、参数、seed 和 clean front RGB；官方 noisy RGB 与 clean replay 均旋转 180° 到同一方向后，再计算 Laplacian、频谱、对比度和残差特征 | 官方 Noise 记录没有噪声算法、参数、seed 或配对 clean frame；官方 clean 对照由匹配 LIBERO replay 重建。复刻只对 front RGB 加噪，wrist 保持 clean | Loss 为 `0.265249 / 0.317787`；只量化观测效果，不恢复官方噪声机制或作用范围 |
| Action / state / trajectory | 192 条参考与 LeRobot 的 float32 action/state 最大误差均为 `0`；复刻完整执行 source actions，仅按 keep mask 保存 post-step observation。6 条视频额外要求任务、setting、action 长度及 float32 bytes SHA-256 全部相同，且 frame `i` 同时取两侧 action index `i` | 相同 action-defined trajectory 不会使随机化资产、渲染参数或相机 pose 自动相同；官方不同导出 episode index 也可能复用同一 action 指纹 | 视频确实反映同一控制 episode，可用于同步观察视觉差异；不能把像素差全归因于动作 |
| 成功轨迹组成 | 72 个复刻 slot 均有冻结 manifest、候选、seed 和成功 HDF5；实际生成账本为 72 成功、1 失败 | 官方失败 attempts、成功筛选逻辑和逐类成功率未公开；本次又只保留成功结果 | 不能由最终 72/72 推断官方或生成器的自然成功率 |
| Depth | 复刻保存 front/wrist 米制 depth、valid mask、near/far 和相机矩阵，几何验证通过 | 三类官方归档均没有 depth 字段或 depth 标定真值 | 不存在官方 depth loss；复刻 depth 只能证明内部几何自洽 |
| 数据集比例与任务覆盖 | 复刻固定 4 suites × 6 settings × 3 slots；每 setting 恰好 12 条，参考池每 setting 恰好 32 条 | 官方数据覆盖 40 个任务和 14,347 条 episode，真实 setting、任务、成功/失败比例没有被复刻保留 | 当前结果是平衡样例集，不是官方训练分布的等比例副本 |

## v2 复刻 / 官方代表性对比视频

左列直接读取 v2 复刻 HDF5，右列直接读取固定 revision 的 LIBERO-Plus precollected TFRecord；视频内明确标为 `REPLICA` / `OFFICIAL`，上下分别为 front 和 wrist。每一对都同时满足任务、setting、action 长度和 `SHA256(contiguous little-endian float32 action bytes)` 相同，最大 action 误差为 `0`。视频保留全部 action timestep，frame `i` 在两侧都对应 action index `i`，不再使用归一化进度。官方 JPEG 按与 reference feature 相同的共享解码规则旋转 180°；这里没有对任一侧做水平或竖直单轴镜像。

| Setting | Replica episode | 对比视频 | 官方记录 | Action 证据 |
|---|---|---|---|---|
| Objects | `libero_goal__objects__slot3` | [`same_action.mp4`](data/libero_plus_rgbd_sample_v2/comparisons/libero_goal__objects__slot3__same_action.mp4) | `ref-25c5ce9404d54f922903` | 107 帧；`9bf695468fb9…`；max error `0` |
| Background | `libero_spatial__background__slot1` | [`same_action.mp4`](data/libero_plus_rgbd_sample_v2/comparisons/libero_spatial__background__slot1__same_action.mp4) | `ref-ba50a9ee54495267de08` | 127 帧；`fdc0170ec83b…`；max error `0` |
| Light | `libero_10__light__slot1` | [`same_action.mp4`](data/libero_plus_rgbd_sample_v2/comparisons/libero_10__light__slot1__same_action.mp4) | Segmentation shard 15 / record 0 | 254 帧；`c80f5afb78c2…`；max error `0` |
| Camera | `libero_spatial__camera__slot1` | [`same_action.mp4`](data/libero_plus_rgbd_sample_v2/comparisons/libero_spatial__camera__slot1__same_action.mp4) | `ref-c3fdc78763ebcdb4573b` | 127 帧；`fdc0170ec83b…`；max error `0` |
| Language | `libero_spatial__language__slot2` | [`same_action.mp4`](data/libero_plus_rgbd_sample_v2/comparisons/libero_spatial__language__slot2__same_action.mp4) | `ref-e2d64b31e462a6067f91` | 133 帧；`9c0249deb6a0…`；max error `0` |
| Noise | `libero_spatial__noise__slot2` | [`same_action.mp4`](data/libero_plus_rgbd_sample_v2/comparisons/libero_spatial__noise__slot2__same_action.mp4) | `ref-5ae8afb6bcf225858d04` | 133 帧；`9c0249deb6a0…`；max error `0` |

其中 5 条直接复用 192 条审计参考；Light 的 192 条抽样参考里没有相同 action，因此从已下载的固定 Segmentation 归档按任务和 action 指纹扫描得到，不把不匹配的 reference 硬凑进视频。其 archive revision、LFS hash、member CRC、record ordinal/offset、raw/frame hash 和 MP4 SHA-256 均见 [`comparison_manifest.json`](data/libero_plus_rgbd_sample_v2/comparisons/comparison_manifest.json)。

## v1 已验证结果

| 项目 | 结果 | 结论 |
|---|---:|---|
| 成功 episode | 72 | 4 suites × 6 settings × 3 slots，覆盖完整 |
| 预览视频 | 24 | 每个 suite × setting 一条 |
| 实际 attempts | 73 | 72 成功、1 失败；失败完整留档 |
| Suite 分布 | 每个 18 条 | 符合样例协议 |
| Setting 分布 | 每类 12 条 | 符合样例协议，不代表官方比例 |
| Source demo 对齐 | 12 个 demo 唯一匹配 | action subsequence 最大误差不超过 `1e-6` |
| 保存 action 误差 | `0` | 保存 action 与 source timestep 精确一致 |
| Replay | 完整 source actions 均执行 | no-op mask 只影响保存，不影响仿真 |
| 相机矩阵互逆误差 | `5.55e-16` | 通过 `1e-6` 阈值 |
| 世界点重投影误差 | `8.04e-14 px` | 通过 1 pixel 阈值 |
| RGB-D | shape、时间索引、near/far、有限值均通过 | Depth 与本次场景和相机几何一致 |
| Setting 隔离 | 全部通过 | Language、Noise、Camera 未越界改变其他字段 |
| 确定性重跑 | 6/6 内容 hash 一致 | 每类 setting 一条完整重跑 |

最终机器可读结果见 [`validation.json`](data/libero_plus_rgbd_sample/reports/validation.json)，汇总见 [`gap_report.md`](data/libero_plus_rgbd_sample/reports/gap_report.md)。

## v1 与官方数据仍无法量化的差异

| 差异 | 当前能保证什么 | 仍缺少什么 | 当前结论 |
|---|---|---|---|
| Front 视角 | 保存逐帧内外参，RGB-D 几何自洽 | 官方逐 episode camera manifest 和 setting 映射 | 不能计算官方 Camera gap |
| Wrist 视角 | 保存逐帧内外参和 robot state | 官方 wrist 标定及可对齐轨迹 | 不能逐帧比较官方画面 |
| 背景纹理 | 参数、资产 ID、XML 和 hash 可复现 | 官方训练纹理池及逐轨迹 texture/material 参数 | 风格接近程度尚未验证 |
| Compounding Objects | 干扰物数量、资产和 pose 可复现，并通过 test 隔离 | 官方资产选择、放置分布和训练 manifest | 这是本地可复现协议，不是官方分布恢复 |
| 光照 | XML 和参数可复现，并通过 test 隔离 | 官方逐 episode 光源参数和 setting 标签 | 明暗及阴影分布尚未校准 |
| Sensor Noise | 五类算法、severity、参数和 clean front RGB 均保存 | 官方噪声类别比例、参数、seed 和作用频率 | 不能计算官方 Noise gap |
| Language | 固定 Qwen revision、prompt、tokenizer hash，rewrite 可复现 | 官方 rewrite 文本、模型和 prompt | 只保证语义等价，不声称恢复官方原句 |
| 成功轨迹组成 | 本次所有 attempts 均记账 | 官方失败 attempts、成功筛选逻辑和逐类成功率 | 不能估计官方成功率或完整训练组成 |
| Depth | 米制深度与本次 RGB、scene、camera 严格一致 | 官方没有 depth 真值 | 不存在官方 depth loss |
| 数据集比例 | 本次固定每类 12 条 | 官方 40 任务的逐类采样与成功分布 | 72 条不能代表完整训练分布 |

## 为什么 v1 calibration/holdout loss 是 unavailable

v1 只下载了 32 个 LeRobot parquet，用于把公开 action 与原始 LIBERO source demo 唯一对齐；当时没有实现固定 RLDS/Camera/Segmentation 归档连接。这是 v1 的实现边界，不是这些归档绝对无法映射。v2 已用完整本地归档恢复六类各 32 条参考，并验证 action/state 精确一致。

因此 v1 没有合法的官方 reference pool，也无法执行 75% calibration / 25% holdout 切分。v1 manifest 明确记录：

- `official_reference_count = 0`
- `calibration_loss = null`
- `holdout_loss = null`
- `confidence = low`
- `fallback = repository generator pool`

这里保留 unavailable 是 v1 的证据边界，不是 v1 生成或验证失败。v2 不再使用 unavailable 硬编码；六类均使用真实 reference IDs、24/8 calibration/holdout 切分并报告有限 loss。公开 Language 文本提供的区分信号较弱，因此该类置信度标为 medium，而不是把缺失值伪装成可用结果。

## v1 已解决与未解决的边界

本次样例已经解决：

- source XML、初态、完整 action replay 和 post-step 保存语义；
- 12 个 source demo 的精确 no-op keep mask；
- RGB-D 格式、相机坐标系、米制 depth 和逐帧 wrist/front 外参；
- 六类 setting 的固定候选、seed、失败记账和 test 隔离；
- 固定 Qwen checkpoint 的确定性 Language rewrite；
- 相同环境下的内容级确定性。

本次样例没有解决：

- 40 个任务及 14,347 条公开 episode 的全量 source 映射；
- 官方逐 episode seed、setting、随机化参数和失败 attempts；
- 官方视觉分布的 calibration/holdout 指标；
- RLDS 或 LeRobot exporter。

## 下一步

若目标只是确认 RGB-D 生成管线是否可用，v1 已经足够；若还要求固定官方参考映射、有限样本校准和 holdout 量化，应使用 v2。两者当前结论均为 GO。

v2 的结论边界仍是 4 个代表任务、192 条官方参考和 72 条生成样例。若要声称覆盖完整官方训练分布，仍需扩展到 40 个任务和 14,347 条 episode，并取得官方逐 episode 随机化参数、失败筛选过程及 Depth 真值（若未来发布）。
