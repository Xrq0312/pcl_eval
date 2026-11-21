# 小基元局部候选与低偏差选模实验

更新时间：2026-11-21
分支：`codex/local-candidate-recall`

## 1. 数据协议

ABCParts-Mini100 继续使用 ParSeNet 训练划分派生数据，不报告为官方测试集。冻结清单
为 `datasets/abcparts_mini100_manifest.json`：前 20 件为开发集，其余 80 件为最终审计
集。审计集只在开发选择冻结后运行。Native 固定种子为 1337。

## 2. 阶段 F1：失败归因

检测器现在导出以下不读取 GT 的候选生命周期快照：

1. `screened_proposals`：通过 pilot 池、第一次在全云评价的候选；
2. `proposals`：重拟合和可观测性门控后的候选；
3. `selected_models`：贪心选模后的候选；
4. `premerge`：点分配和迭代重拟合后的实例；
5. `final`：连通处理后的最终实例。

离线评测使用检测器实际输出的系数、阈值和法向重建支持集。候选阶段使用同类型
IoU `>=0.30`，最终检测使用同类型 IoU `>=0.50`。GT 只参与离线归因。

`candidate_not_generated` 的准确含义是：没有候选通过最小采样、pilot 排名并进入全云
筛选且达到 IoU 0.30。它不能进一步区分“从未采中”和“在 pilot 池被淘汰”，后续若
需要可增加采样命中日志。

ABCParts 开发 20 件共有 288 个目标实例：

| 阶段 | screened 候选召回 | accepted 候选召回 | selected 召回 | 最终 Recall |
|---|---:|---:|---:|---:|
| `global_ransac` | 24.7% | 24.3% | 24.3% | 19.8% |
| `joint_opt` | 19.4% | 19.1% | 19.1% | 16.3% |

失败归因：

| 阶段 | 候选未生成 | 候选过滤 | 分配/重拟合损失 | 检出 |
|---|---:|---:|---:|---:|
| `global_ransac` | 75.3% | 0.35% | 4.5% | 19.8% |
| `joint_opt` | 80.6% | 0.35% | 2.8% | 16.3% |

按关键子集：

| 子集 | global 候选未生成 | joint 候选未生成 |
|---|---:|---:|
| 小实例 `<=100`，n=174 | 98.3% | 98.3% |
| 圆柱，n=55 | 92.7% | 89.1% |
| complex，n=203 | 81.3% | 83.7% |

因此下一步首先提高局部候选召回。现有候选一旦进入 accepted pool，当前贪心选模在
IoU 0.30 口径下几乎没有额外损失；低偏差选模仍需在局部生成器成功增加小候选后再
评估，避免优化一个当前并非主导的阶段。

复现：

```powershell
& .\tools\run_python.ps1 tools/run_native_ablation.py `
  --exe build-native/Release/native_primitive_fit.exe `
  --samples samples/abcparts_mini100 `
  --results results/local_candidate_dev20_baseline `
  --stages 'global_ransac,joint_opt' --distance 0.02 --min-points 80 `
  --manifest datasets/abcparts_mini100_manifest.json `
  --split development --force

& .\tools\run_python.ps1 tools/summarize_failure_attribution.py `
  --results-root results/local_candidate_dev20_baseline `
  --samples-root samples/abcparts_mini100 `
  --output results/local_candidate_dev20_baseline/failure_attribution.json
```

## 3. 阶段 F2：局部多尺度候选

新增的 `adaptive_local` 是独立可选实验阶段，未改变 `global_ransac`、`global_local`、
`joint_opt` 的执行路径。它复用现有 C++ 稳健拟合模块，而没有新增圆柱拟合公式：

1. 使用固定种子在点云中抽取 2048 个 anchor；
2. 轮换使用 24、48、96、192、384 点 kNN 邻域以及平面、球面、圆柱三种模型；
3. 平面使用局部重拟合；球面和圆柱复用 LS、TLS、IRLS、稳健点对/point-only 候选；
4. 将候选限制在 anchor 的 `1.25 * kNN radius` 空间范围内，避免小面立即扩展到远处
   共面或同轴曲面；
5. 实验阶段的最低最终支持从 80 降为 20，Native 阶段仍使用 80。

“自适应”目前仅指多尺度邻域与各邻域的空间半径，不代表已经实现基于噪声、曲率或
局部点间距的连续尺度估计。这个限制必须和实验结论一起保留。

ABCParts 开发 20 件的微平均结果如下。Precision 的分母是最终预测实例数，Recall 和
mIoU 的分母是 288 个 GT 目标实例。

| 阶段 | 预测数 | Recall | Precision | mIoU | 算法时间/件 |
|---|---:|---:|---:|---:|---:|
| `global_ransac` | 147 | 19.79% | 38.78% | 18.02% | 0.062 s |
| `joint_opt` | 163 | 16.32% | 28.83% | 15.36% | 0.518 s |
| `adaptive_local` | 670 | **21.53%** | 9.25% | **20.19%** | 0.345 s |

`adaptive_local` 的 accepted 候选召回为 27.08%，selected 为 26.39%，最终为 21.53%。
相对 `global_ransac`，最终多检出 5 个实例，但预测数增加 523 个，Precision 下降
29.53 个百分点。按实例大小看，小实例 `<=100` 的 accepted 召回只有 3.45%，最终
Recall 只有 2.30%；按类型看，圆柱 accepted 召回 10.91%，最终 Recall 7.27%。因此
2048 个均匀随机 anchor 并没有解决小实例和圆柱候选覆盖问题，且产生了大量局部碎片。

## 4. 阶段 F3：低支持偏差选择

`balanced_select` 固定使用 F2 候选，只替换选择与竞争规则：以单位支持得分和归一化
RMSE 排序，加入 30% 新支持率、同类/跨类重叠去重，取消分配时的大支持先验，并使用
更严格的重拟合 RMSE 门控。它同样是非默认实验阶段。

| 阶段 | accepted 召回 | selected 召回 | 最终 Recall | Precision | mIoU |
|---|---:|---:|---:|---:|---:|
| `adaptive_local` | 27.08% | 26.39% | 21.53% | 9.25% | 20.19% |
| `balanced_select` | 27.08% | 26.04% | 19.79% | 8.39% | 17.96% |

低偏差规则没有挽回候选生成缺失，反而让分配/重拟合损失从 14 个实例增加到 18 个。
这验证了 F1 的判断：当前选择阶段不是主导瓶颈；在候选池仍缺少正确小实例的情况下，
改变排序先验主要会重新排列错误候选。

## 5. 跨数据集验证

所有参数只在 ABCParts 开发 20 件确定。随后不调参地运行 Fit4CAD 官方测试适配样本
23 件、SHREC2022 固定测试子集 30 件和 BAGSFit 固定前 10 件。Fit4CAD/SHREC 使用
自动距离阈值，BAGSFit 使用 0.02。以下是场景检测微平均，属于当前固定子集的内部验证，
不冒称完整数据集官方成绩。

| 数据 | 阶段 | GT/预测 | Recall | Precision | mIoU | 时间/件 |
|---|---|---:|---:|---:|---:|---:|
| Fit4CAD-23 | `joint_opt` | 487/279 | 22.59% | **39.43%** | 23.47% | 0.647 s |
| Fit4CAD-23 | `adaptive_local` | 487/511 | 23.00% | 21.92% | 26.57% | 0.559 s |
| Fit4CAD-23 | `balanced_select` | 487/748 | **26.28%** | 17.11% | **29.28%** | 0.543 s |
| SHREC2022-30 | `joint_opt` | 30/75 | **80.00%** | **32.00%** | **78.51%** | **0.266 s** |
| SHREC2022-30 | `adaptive_local` | 30/199 | 73.33% | 11.06% | 77.10% | 0.427 s |
| SHREC2022-30 | `balanced_select` | 30/85 | 76.67% | 27.06% | 71.96% | 0.401 s |
| BAGSFit-10 | `joint_opt` | 107/441 | 26.17% | **6.35%** | **31.36%** | **3.099 s** |
| BAGSFit-10 | `adaptive_local` | 107/457 | **27.10%** | 6.35% | 30.49% | 6.093 s |
| BAGSFit-10 | `balanced_select` | 107/477 | 14.02% | 3.14% | 15.15% | 5.653 s |

圆柱参数误差只对成功且参数 GT 可用的匹配计算，不能与检出率分开解读：

| 数据 | 阶段 | 圆柱匹配数 | 轴向误差 | 半径相对误差 |
|---|---|---:|---:|---:|
| Fit4CAD-23 | `joint_opt` / `adaptive_local` / `balanced_select` | 44 / 33 / 42 | 0.539° / 0.944° / 0.694° | 0.055% / 0.381% / 0.173% |
| SHREC2022-30 | `joint_opt` / `adaptive_local` / `balanced_select` | 8 / 7 / 8 | 0.221° / 0.195° / 0.200° | 0.177% / 0.179% / 0.269% |
| BAGSFit-10 | `joint_opt` / `adaptive_local` / `balanced_select` | 11 / 10 / 6 | 0.584° / 0.673° / 0.372° | 1.741% / 9.636% / 3.121% |

Fit4CAD 表明较低支持门槛可能提高候选覆盖，但 Precision 损失很大；SHREC 上两项新方法
均退化；BAGSFit 上自适应候选只增加 0.93 个百分点 Recall，却约翻倍耗时并恶化圆柱
半径误差。收益没有跨数据集稳定性。

## 6. 决策与审计状态

- **纳入工具链：** F1 失败归因系统和微平均汇总；它不读取 GT 参与检测。
- **保留为实验：** `adaptive_local`、`balanced_select`，便于继续研究和复现实验，但不
  替换 Native，不作为推荐配置。
- **不进入 Mini80 审计：** 两个方案在开发集均未通过 Precision、跨数据集稳定性和
  CPU 成本门槛。为避免用审计集反复挑方案，冻结的 80 件仍未运行、未参与任何调参。
- **Native 等价性：** 用最终可执行文件重跑开发 20 件后，`global_ransac` 和
  `joint_opt` 的 40 组 `out_labels.csv`、`out_models.csv` 与 F1 基线逐文件 SHA-256
  一致；新增代码没有改变基线路径输出。

当前真正未解决的问题仍是候选覆盖：均匀随机 anchor 对 `<=100` 点实例的命中率太低，
而降低支持门槛会制造大量碎片。下一轮应优先研究由边界、曲率/ASO、法向补偿低秩稳定性
或几何投票引导的 anchor/邻域，而不是继续放松最终阈值或单独调整候选排序。

复现实验：

```powershell
& .\tools\run_python.ps1 tools/run_native_ablation.py `
  --exe build-native/Release/native_primitive_fit.exe `
  --samples samples/abcparts_mini100 `
  --results results/local_candidate_dev20_adaptive2048 `
  --stages 'adaptive_local,balanced_select' --distance 0.02 `
  --min-points 80 --small-min-points 20 --adaptive-local-anchors 2048 `
  --manifest datasets/abcparts_mini100_manifest.json `
  --split development --force
```
