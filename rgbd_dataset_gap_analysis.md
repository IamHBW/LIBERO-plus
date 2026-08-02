# LIBERO-Plus RGB-D v1/v2 差异与验证报告

## 结论

v1 的 72 条 success-selected RGB-D 样例已通过工程验证（GO）。v2 已恢复 192 条官方参考的唯一映射，冻结 216 次尝试并生成 72 条 HDF5、24 个预览；完整验证和 6/6 确定性复跑均通过，结论为 **GO**。v1 文件未被修改。

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
| 映射置信度 | 192 条 high | revision、volume、member、record、raw/frame hash 均可追踪 |
| 扫描范围 | Camera 256；Segmentation 1024 shards | 使用本地完整归档；没有建立 14,347 条持久索引 |
| Calibration probes / attempts | 576 / 216 | 每槽 8 个 probe，冻结最低 loss 的 3 次 retry |
| v2 HDF5 / 预览 | 72 / 24 | 4 suites × 6 settings × 3 slots，覆盖完整 |
| 实际 attempts | 72 | 全部成功，失败为 0 |
| Suite / setting 分布 | 每 suite 18；每 setting 12 | 与冻结协议一致 |
| 确定性重跑 | 6/6 | 每类 setting 一条内容 hash 一致 |
| 最终结论 | **GO** | 六类 calibration / holdout loss 均为有限数值 |

| Setting | 样例 | 官方参考（calibration/holdout） | Calibration loss | Holdout loss | 置信度 |
|---|---:|---:|---:|---:|---|
| Objects | 12 | 32（24/8） | 0.894728 | 0.898675 | high |
| Background | 12 | 32（24/8） | 12.9945 | 12.6502 | high |
| Light | 12 | 32（24/8） | 0.116396 | 0.254526 | high |
| Camera | 12 | 32（24/8） | 4.89106 | 4.94660 | high |
| Language | 12 | 32（24/8） | 97222.2 | 97222.2 | medium |
| Noise | 12 | 32（24/8） | 0.742547 | 0.782892 | high |

机器可读映射见 [`reference_mapping.jsonl`](data/libero_plus_rgbd_sample_v2/evidence/reference_mapping.jsonl)，参考内容见 [`official_references.hdf5`](data/libero_plus_rgbd_sample_v2/evidence/official_references.hdf5)，数值结果见 [`calibration.json`](data/libero_plus_rgbd_sample_v2/reports/calibration.json)，最终验收见 [`validation.json`](data/libero_plus_rgbd_sample_v2/reports/validation.json) 和 [`gap_report.md`](data/libero_plus_rgbd_sample_v2/reports/gap_report.md)。

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
