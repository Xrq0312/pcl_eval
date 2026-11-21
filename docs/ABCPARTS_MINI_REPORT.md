# ABCParts-Mini100 数据与评测报告

更新时间：2026-11-20  
选择协议：`abcparts_train_mini_stratified_v2`  
固定种子：`20261120`

## 1. 来源核对与限制

[HPNet 官方仓库](https://github.com/SimingYan/HPNet)只提供约 69 GB 的完整预处理
ABCParts 下载；其预处理会将 ParSeNet 的单体 split HDF5 拆成逐零件 HDF5，并增加
`T_param`。HPNet 的加载器读取 `points`、`normals`、`labels`、`prim` 和 `T_param`，
还会把不超过 100 点的实例从训练标签中设为背景。

[ParSeNet 数据说明](https://github.com/Hippogriff/parsenet-codebase/blob/master/readme_data.md)
描述的是 `train_data.h5`、`val_data.h5`、`test_data.h5` 三个单体文件以及对应 ID 列表；
官方论文使用 24k/4k/4k 的 train/validation/test 划分。其
[下载脚本](https://raw.githubusercontent.com/Hippogriff/parsenet-codebase/master/download_dataset.sh)
中的 `data.zip` 下载已经被注释，并没有可按单个零件下载的官方接口。

本次用户提供的 `data/train_data.h5` 经只读核对为：

| 字段 | shape | dtype |
|---|---:|---|
| `points` | 24000 × 10000 × 3 | float32 |
| `normals` | 24000 × 10000 × 3 | float32 |
| `labels` | 24000 × 10000 | int16 |
| `prim` | 24000 × 10000 | int16 |

文件大小 6,720,004,096 bytes，无压缩、无 `T_param`，也没有随文件提供
`train_ids.txt`。因此：

- 本集合是 **ParSeNet 官方训练划分派生的可复现压力测试集**，不是官方 test 成绩；
- 样本 ID 使用可恢复的 HDF5 行号，如 `train_02172`；不能声称它是原始 ABC CAD ID；
- 点、法向、实例标签和曲面类型为官方数据；
- 平面、球面、圆柱参数为从官方点/法向拟合的 `derived reference`，不是原始 CAD
  analytic GT。

本阶段没有下载 HPNet 69 GB 数据，也没有用合成零件替代官方样本。

## 2. 固定选择协议

选择器为 `tools/prepare_abcparts_mini.py`，完整清单和每个样本哈希在
`datasets/abcparts_mini100_manifest.json`。

选择过程只扫描 `labels/prim`，不读取检测结果或参数拟合质量：

1. 过滤至少含一个平面、球面或圆柱的训练零件；
2. 以全训练集实例数的 33%/67% 分位数定义复杂度：simple `<=8`、medium
   `9–22`、complex `>22`；
3. 固定选择 20 simple、40 medium、40 complex；
4. 贪心覆盖平面、球面、圆柱、小实例、非目标曲面和多目标实例，哈希种子只用于稳定
   tie-break；
5. 顺序按 4/8/8 的比例交织，因此 `mini_000`–`mini_019` 是固定 smoke20，完整前缀
   不随文件系统排序变化。

最终覆盖：100/100 含平面、100/100 含圆柱、75/100 含球面、86/100 含不超过
100 点的目标实例、100/100 含非目标干扰曲面。

每个适配样本保留原始 10,000 点，并输出 `points.txt`、`normals.txt`、
`gt_labels.csv`、`gt_primitives.json` 和 `metadata.json`。非目标 cone、open/closed
spline、extrusion、revolution 全部保留，未删除或改成背景。

## 3. 参数参考质量门控

ParSeNet HDF5 不含解析参数。适配器使用官方点/法向拟合参考参数，但只有满足以下条件的
实例才标记 `has_params=true`：

- 95% 曲面残差 `<=0.01`；
- 球面/圆柱半径在 `(0,10]`，与 HPNet 预处理的超大半径拒绝规则一致；
- 所有参数有限且实例至少 6 点。

100 个零件中可靠 derived reference 数量为：平面 885、球面 88、圆柱 146。
参数误差只在这些实例且检测类型正确、IoU `>=0.5` 时统计。

## 4. 评测协议

### 4.1 完整场景自动检测

Native 输入完整 10,000 点，不读取 GT 或官方法向。冻结配置：

- `distance=0.02`
- `min_points=80`
- `global_ransac` 与 `joint_opt`
- 实例匹配采用 Hungarian one-to-one IoU；Recall/Precision 阈值为 IoU `>=0.5`；
- 实例 mIoU 对所有目标 GT 求均值，未匹配实例计 0。

纯点圆柱投票同样读取完整场景 XYZ，固定 96 个方向、每方向 128 个三点假设、
`minimum_support=40`。它是 cylinder-only proposal 实验，不与三类型 Native 指标混称。

### 4.2 Oracle 实例拟合

GT 实例标签先切出平面、球面、圆柱块；至少 20 点的块进入实验，单块最多固定采样
2048 点。法向由 PCL PCA kNN 估计，不读取 GT 类型参与拟合。模型竞争同时拟合三种
类型。该结果只衡量候选初始化和类型选择，不能报告为场景 Recall、Precision 或 IoU。

## 5. 20 零件快速验证

smoke20 含 288 个目标 GT 实例和 55 个圆柱：

| 方法 | Recall@0.5 | Precision@0.5 | 实例 mIoU | 算法时间/零件 |
|---|---:|---:|---:|---:|
| Native global_ransac | 19.8% | 38.8% | 18.0% | 0.056 s |
| Native joint_opt | 16.3% | 28.8% | 15.4% | 0.511 s |
| 纯点圆柱投票 | 1.8% | 0.26% | 5.2% | 1.295 s |

Oracle 块中纯 BIC 为 63.6%，加入分半稳定性后为 80.7%。快测没有崩溃、标签长度
错误或非有限 Native 参数，因此按原配置运行全部 100 个，没有根据 smoke 调阈值。

## 6. 100 零件完整结果

集合包含 2,927 个曲面实例，其中目标实例 1,902 个：平面 1,090、圆柱 608、球面
204。非目标实例包括 closed spline 284、open spline 376、cone 305、extrusion 49、
revolution 11。目标实例中 1,037 个不超过 100 点。

### 6.1 完整场景 Native

以下是按实例数汇总的微平均，不是每场景宏平均：

| 方法 | 预测实例 | Recall@0.5 | Precision@0.5 | 实例 mIoU | 算法时间/零件 |
|---|---:|---:|---:|---:|---:|
| global_ransac | 857 | **15.3%** | **34.0%** | **15.2%** | **0.061 s** |
| joint_opt | 860 | 14.2% | 31.5% | 14.1% | 0.522 s |

`joint_opt` 在圆柱和大实例上更好，但在平面、中等实例和复杂零件上损失更多，整体没有
超过 `global_ransac`。

按曲面类型：

| 方法 | 平面 Recall / mIoU | 球面 Recall / mIoU | 圆柱 Recall / mIoU |
|---|---:|---:|---:|
| global_ransac | 21.2% / 18.7% | 17.6% / 20.3% | 3.9% / 7.3% |
| joint_opt | 15.0% / 13.5% | **23.5% / 24.3%** | **9.9% / 11.7%** |

按实例点数：

| 方法 | `<=100` Recall / mIoU | `101–500` | `>500` |
|---|---:|---:|---:|
| global_ransac | 0.19% / 0.41% | **26.1% / 24.3%** | 44.8% / 46.6% |
| joint_opt | 0.19% / 0.26% | 16.3% / 16.1% | **54.3% / 53.5%** |

按零件复杂度：

| 方法 | simple | medium | complex |
|---|---:|---:|---:|
| global_ransac Recall / mIoU | 37.6% / 35.7% | **27.5% / 24.7%** | **10.2% / 11.1%** |
| joint_opt Recall / mIoU | **48.4% / 46.7%** | 26.3% / 24.4% | 8.4% / 8.8% |

### 6.2 圆柱参数误差

只统计可靠 derived reference 且类型正确、IoU `>=0.5` 的匹配：

| 方法 | 匹配数 | 轴向角均值/中位数 | 半径相对误差均值/中位数 |
|---|---:|---:|---:|
| global_ransac | 11 | 0.228° / 0.101° | 6.02% / 0.0033% |
| joint_opt | 20 | 0.598° / 0.205° | 0.558% / 0.0067% |

global_ransac 的均值被少数错误轴线/半径匹配拉高；中位数很小。样本数只有 11/20，
不能把参数精度推广到未检出的 608 个圆柱。

### 6.3 Oracle 候选与模型竞争

1,496 个不少于 20 点的目标实例块：

| 竞争项 | 分类准确率 | 拒判率 |
|---|---:|---:|
| 纯 trimmed BIC | 68.6% | 6.1% |
| +分半参数稳定性 | **78.7%** | 3.6% |
| +ASO 软先验 | 73.9% | 4.5% |
| 稳定性+ASO | 78.6% | 3.8% |

纯 BIC 按类型为：平面 77.0%、球面 100%、圆柱 47.3%；按大小为：小实例 43.9%、
中等实例 78.2%、大实例 99.7%。ABCParts 上稳定性项有明显收益，但此前 Fit4CAD 上有
下降，因此仍作为实验开关，不据单一数据集修改 Native。

质量门控参数参考上的候选成功率：

- 球面 `n=83`：LS/TLS/IRLS 均 14.5%，纯点代数球 100%；
- 圆柱 `n=146`：LS 34.9%、TLS 35.6%、Pair-IRLS 43.2%；
- 平均单候选开销：TLS 圆柱约 0.006 ms、Pair-IRLS 约 1.06 ms。

### 6.4 完整场景纯点圆柱投票

在 608 个 GT 圆柱上：

- Recall@0.3 5.76%，候选 Precision@0.3 3.14%；
- Recall@0.5 2.63%，候选 Precision@0.5 0.99%；
- 实例 mIoU 5.61%，平均 1.267 s/零件；
- 中/小圆柱 Recall@0.5 均为 0，大圆柱为 10.4%。

该原型不适合 ABCParts 复杂场景，继续保持“不接入 Native”的结论。

## 7. 失败原因

1. **小实例是首要瓶颈**：54.5% 的目标实例不超过 100 点；Native 使用
   `min_points=80`，加上区域生长/模型支持要求后，small Recall 仅 0.19%。
2. **圆柱候选不足**：global_ransac 圆柱 Recall 3.9%，joint_opt 虽升到 9.9%，仍远
   低于平面和球面。短弧和大半径圆柱容易被平面/球面竞争掉。
3. **复杂零件竞争严重**：complex Recall 只有 10.2%/8.4%；同一场景多达 50 个曲面，
   大模型吸收、碎片候选和未分配点同时出现。
4. **假候选多**：global_ransac/joint_opt 分别有 564/570 个预测在 IoU 0.5 下未匹配；
   joint_opt 另有 18 个预测完整吸收非目标实例。
5. **全局投影不适合拥挤 CAD**：纯点圆柱投票产生 1,815 个候选，但 IoU 0.5 命中仅
   18 个；平面、球面和 spline 的投影圆带造成大量竞争模型。

## 8. 复现

```powershell
# 首次固定选择并适配；不会下载外部数据
& .\tools\run_python.ps1 tools/prepare_abcparts_mini.py `
  --input data/train_data.h5 `
  --manifest datasets/abcparts_mini100_manifest.json `
  --out-root samples/abcparts_mini100 `
  --count 100 --smoke-count 20 --seed 20261120

# 以后严格复用已冻结 ID
& .\tools\run_python.ps1 tools/prepare_abcparts_mini.py `
  --input data/train_data.h5 `
  --manifest datasets/abcparts_mini100_manifest.json `
  --out-root samples/abcparts_mini100 --reuse-manifest

& .\tools\run_python.ps1 tools/run_native_ablation.py `
  --exe build-native/Release/native_primitive_fit.exe `
  --samples samples/abcparts_mini100 `
  --results results/abcparts_mini100/native `
  --stages 'global_ransac,joint_opt' --distance 0.02 --min-points 80

& .\tools\run_python.ps1 tools/run_feature_probe.py `
  --root samples/abcparts_mini100 `
  --output results/abcparts_mini100/oracle_candidates.json `
  --normal-k 64 --aso-k 64 --max-points 2048 --min-points 20

& .\tools\run_python.ps1 tools/run_cylinder_voting_probe.py `
  --root samples/abcparts_mini100 `
  --output results/abcparts_mini100/cylinder_voting.json `
  --max-points 10000 --directions 96 --hypotheses 128 --minimum-support 40

& .\tools\run_python.ps1 tools/summarize_abcparts_mini.py `
  --samples-root samples/abcparts_mini100 `
  --native-root results/abcparts_mini100/native `
  --oracle results/abcparts_mini100/oracle_candidates.json `
  --voting results/abcparts_mini100/cylinder_voting.json `
  --output results/abcparts_mini100/summary_stratified.json
```

`samples/` 与 `results/` 是可再生输出并受 `.gitignore` 管理；选择 manifest、适配器、评测
脚本和本报告进入版本管理。
