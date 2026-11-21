# 小基元局部候选与低偏差选模实验

更新时间：2026-11-20  
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

## 3. 下一阶段

F2 实现自适应局部、多尺度、低最低支持候选，并只在上述 20 件开发集选择一次配置。
F3 在 F2 候选固定后比较原贪心选择与低支持偏差选择。最终配置再运行 80 件冻结审计
集以及 BAGSFit、Fit4CAD、SHREC2022。
