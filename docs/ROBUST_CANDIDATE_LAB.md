# 稳健候选生成与多模型竞争实验报告

更新时间：2026-11-20  
分支：`codex/robust-candidate-competition`

## 1. 范围与数据泄漏约束

本阶段只增加独立实验库、探针、测试和评测脚本，没有修改 `primitive_core`、
`native_primitive_fit` 或 Native 检测路径。实验严格分成两类：

1. `oracle_instance_patch`：GT 实例标签只用于切出单个实例块。候选拟合和模型
   竞争不知道真实类型，GT 参数只在拟合完成后计算误差。该结果不是场景检测召回率，
   也不是实例 IoU。
2. `full_scene_point_only_candidate_detection`：候选生成器只读取完整场景 XYZ，
   不读取法向、类型、GT 或区域生长结果。检测结束后才读取 GT 做 IoU 和参数匹配。

Native 仍是唯一生产基线。本报告中的实验模块尚未接入 Native。

## 2. 论文与公开实现核对

### 2.1 Progressive-X / Multi-X

- [Progressive-X](https://openaccess.thecvf.com/content_ICCV_2019/papers/Barath_Progressive-X_Efficient_Anytime_Multi-Model_Fitting_Algorithm_ICCV_2019_paper.pdf)
  的关键不是一次选出一个最优模型，而是维护 putative/active hypotheses，交替执行
  候选生成、快速拒绝、标签优化、模型整合，并保留 outlier 类。
- [Multi-X](https://openaccess.thecvf.com/content_ECCV_2018/papers/Daniel_Barath_Multi-Class_Model_Fitting_ECCV_2018_paper.pdf)
  将数据项、空间一致性和模型复杂度放在同一个能量中，并支持不同模型类别共同竞争。

本阶段只实现了适合单实例块消融的最小版本：平面、球面和圆柱共享同一残差尺度，
以 90% trimmed BIC 比较模型，并允许低置信度拒判。它不是 Progressive-X/PEARL 的
完整复现，也没有把 oracle 块结果误称为多实例场景检测。

### 2.2 两条圆柱检测路线

- Bergamasco 等人的
  [Cylinders extraction in non-oriented point clouds as a clustering problem](https://iris.unive.it/bitstream/10278/3734123/1/1-s2.0-S0031320320302466-main.pdf)
  不要求输入法向：随机切平面、在切面点集中稳健拟合椭圆，由椭圆轴长恢复半径和两个
  可能轴向，再对候选做兼容性聚类和 evolutionary-game 选择。
- Araújo 与 Oliveira 的
  [Connectivity-based Cylinder Detection 官方 GPL 实现](https://github.com/abner-math/CylinderDetection)
  虽然支持任意圆柱朝向，但并非无方向/无法向方法。实现会采样半球方向，按
  `abs(n·axis)` 过滤点，计算 3D 连通分量，投影后检测圆并用法向与角覆盖验证。

因此，本项目没有把 Connectivity-based 方法错误描述为“完全不使用法向”。当前纯点
原型借鉴其“方向采样—投影—圆验证—连通/覆盖验证”结构，但候选生成只用 XYZ；局部
PCA 法向只作为候选生成后的弱验证，不是输入或预分割条件。

## 3. 实现

### 3.1 稳健低秩候选

独立库：`robust_candidates`。

对球面使用 `q_i = p_i - R n_i` 应收敛到球心；对圆柱将点和法向投影到轴向正交平面，
补偿点应收敛到轴线。比较四种初始化：

- `normal_ls`：已有低秩最小二乘，作为原始对照；
- `normalized_tls`：在无量纲化的法向差/位置差二维散布矩阵上取 TLS 主方向；
- `pair_irls`：由点对约束产生半径种子，取中位数后用 Huber 权重迭代；
- `point_only`：球面的代数最小二乘，不使用法向，并对平面退化和超大半径解做条件数
  检查。

圆柱轴向由 `Σ n_i n_i^T` 的最小特征向量初始化。该二阶矩对逐点法向翻转不变，避免
了中心化法向协方差在翻转时失效。它仍然依赖法向方向精度，所以不等价于纯点圆柱拟合。

### 3.2 多模型竞争

每个 oracle 实例块同时生成：

- 3 参数平面；
- 4 参数球面；
- 5 参数圆柱。

基础分数为：

`BIC = n_keep log(trimmed_MSE / patch_scale²) + k log(n_keep)`。

只比较不同模型类型间的次优分数；同一类型的重复候选不会虚假压低置信度。若最佳归一化
RMSE 大于 0.08，或每点 BIC margin 小于 0.02，则返回 `rejected`。

两个可选软项均做了独立消融：

- 参数稳定性：偶/奇点分半重拟合，按中心/轴线/半径的无量纲差异连续加罚；
- 多尺度 ASO：把平坦度、两主曲率平衡度和小主曲率比例转成连续代价，只用于打破接近
  的残差，不作硬类型判定。

已有曲率边界残差继续作为软连接代价接口。当前 oracle 实验没有实例邻接图，纯点投票也
没有可靠法向图，因此本阶段没有伪造“边界项”消融结果。

### 3.3 无预分割纯点圆柱候选

独立库：`cylinder_voting`。流程为：

1. Fibonacci 半球采样轴向；
2. 将完整场景投影到轴向正交平面；
3. 三点圆 RANSAC 生成候选并以代数圆重拟合；
4. 检查支持数、角覆盖、轴向跨度和径向残差；
5. 从 XYZ 邻域临时估计局部 PCA 法向，只作径向一致性验证；
6. 在同一支持集上让圆柱与平面、球面竞争；
7. 按轴向、轴线、半径和支持重叠做非极大抑制。

该实现是用于验证方向投影/投票可行性的 CPU 原型，不是 Bergamasco 椭圆切片算法的复现。
它没有使用 GT、预估法向或区域生长来生成候选。

## 4. 指标

oracle 参数候选成功阈值：

- 球面：半径相对误差 `< 5%`，球心误差/半径 `< 10%`；
- 圆柱：半径相对误差 `< 5%`，轴向误差 `< 2°`，轴线偏移/半径 `< 10%`。

场景投票报告两级召回：

- candidate recall：候选与 GT 点集 IoU `>= 0.30`；
- strict recall：再要求半径误差 `< 10%`、轴向误差 `< 5°`、轴线偏移 `< 20%`；
- 假候选：与任何 GT 圆柱的 IoU `< 0.10`。

## 5. 实验结果

### 5.1 合成 oracle 块：稳健低秩

共 80 个 patch、231 条方法记录。主要结果：

| 条件 | LS | TLS | Pair-IRLS | Point-only |
|---|---:|---:|---:|---:|
| 球面覆盖/密度，12 cases | 100% | 100% | 100% | 100% |
| 球面法向噪声/翻转，12 cases | 33.3% | 33.3% | 50.0% | **100%** |
| 球面 5–20% 点离群，3 cases | 0% | 0% | **100%** | 0% |
| 圆柱覆盖/密度，15 cases | 80.0% | **100%** | 80.0% | — |
| 圆柱法向噪声/翻转，12 cases | 25.0% | 33.3% | **41.7%** | — |
| 圆柱 5–20% 点离群，3 cases | 0% | 33.3% | **66.7%** | — |
| 圆柱短轴向/60°弧，5 cases | 0% | 0% | 0% | — |

结论：Pair-IRLS 对离群点和部分法向翻转有效，但约比 TLS 慢 2–3 个数量级；纯点球拟合
不受法向翻转影响，却会被点离群严重破坏。短圆柱段中轴向虽然稳定，半径和轴线位置仍不
可靠，不能把“补偿点接近一条线”解释为参数一定可靠。

### 5.2 真实数据 oracle 块：候选召回率

| 数据集/类型 | LS | TLS | Pair-IRLS | Point-only |
|---|---:|---:|---:|---:|
| BAGSFit 球面，n=19 | 36.8% | 26.3% | **52.6%** | 31.6% |
| BAGSFit 圆柱，n=31 | **22.6%** | 16.1% | 19.4% | — |
| SHREC2022 球面，n=10 | 70.0% | **90.0%** | **90.0%** | **90.0%** |
| SHREC2022 圆柱，n=10 | 80.0% | **90.0%** | **90.0%** | — |
| Fit4CAD 球面，n=2 | 50.0% | 50.0% | 50.0% | **100%** |
| Fit4CAD 圆柱，n=163 | 39.9% | 42.3% | **42.9%** | — |

运行时间上，TLS 通常为 `0.006–0.026 ms/实例候选`，Pair-IRLS 为
`1.06–5.34 ms/实例候选`。真实跨数据集结果不支持“IRLS 总是优于 LS”：BAGSFit
圆柱上 LS 仍最好；TLS 是更稳定的低成本补充候选。

### 5.3 模型竞争消融

准确率把拒判计为错误，因而同时报告拒判率。

| 数据集 | 纯 BIC | +稳定性 | +ASO 软先验 | 两者组合 |
|---|---:|---:|---:|---:|
| BAGSFit，102 blocks | 73.5% | 74.5% | 73.5% | **75.5%** |
| SHREC2022，30 blocks | 90.0% | 90.0% | 90.0% | 90.0% |
| Fit4CAD，346 blocks | **82.1%** | 81.5% | 81.8% | 81.5% |

三数据集合计，纯 BIC 与组合项均为 386/478 正确；组合项没有净收益，并且降低
Fit4CAD。稳定性和 ASO 权重默认关闭，保留为实验参数，不进入后续 Native 改造建议。
纯 BIC 竞争相对之前固定特征规则在 SHREC2022 从 60% 提升到 90%，在 BAGSFit 从
39.2% 提升到 73.5%；Fit4CAD 的固定规则为 86.7%，纯 BIC 为 82.1%，说明两者仍需
跨数据集校准，不能只看单一数据集。

### 5.4 完整场景纯点圆柱投票

统一使用 3000 点、96 个方向、每方向 128 个三点假设；这是候选原型，不是 Native。

| 数据集 | GT 圆柱 | IoU≥0.30 候选召回 | strict 参数召回 | 假候选比例 | CPU/场景 |
|---|---:|---:|---:|---:|---:|
| SHREC2022 | 10 | **80.0%** | 30.0% | 46.1% | 442 ms |
| BAGSFit 10 scenes | 30 | 26.7% | 0.0% | 60.0% | 603 ms |
| Fit4CAD 23 scenes | 253 | 5.1% | 1.2% | 58.2% | 467 ms |

失败原因很明确：全局方向投影对单一、覆盖充分的圆柱有较高候选召回，但在多曲面拥挤
场景中，随机平面/球面带状支持也会形成圆投影；全局 3000 点下小圆柱的三点同实例采样
概率又过低。局部验证降低了早期烟测的候选数，但没有把跨数据集假阳性降到可集成水平。
该模块保留作负结果和后续“局部切片椭圆 + 假设聚类”的起点，不接入 Native。

## 6. 决策

当前明确值得保留到下一阶段的模块：

1. **TLS 作为廉价并行候选**：SHREC 和 Fit4CAD 圆柱有收益，CPU 开销很小；
2. **Pair-IRLS 作为离群点备用候选**：球面收益最稳定，但必须由残差/稳定性选择，不能
   无条件替代 LS；
3. **纯 BIC 多模型竞争与拒判框架**：明显优于固定规则的两个数据集，但 Fit4CAD 仍低于
   固定规则，需在完整标签优化中继续验证；
4. **纯点投影候选原型保留但不集成**：召回和误检不满足要求；
5. **稳定性和 ASO 软先验默认关闭**：当前没有跨数据集净收益；
6. **曲率边界只保留为未来标签图的软边代价**，不改成硬切分阈值。

## 7. 复现

```powershell
cmake --preset release -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake
cmake --build --preset release -j 4
ctest --preset release --output-on-failure

.\build-native\Release\robust_candidate_synthetic.exe `
  --output results\candidates\synthetic.csv
& .\tools\run_python.ps1 tools/summarize_candidate_experiment.py `
  --input results/candidates/synthetic.csv `
  --output results/candidates/synthetic_summary.json

& .\tools\run_python.ps1 tools/run_feature_probe.py `
  --root samples/shrec2022_test30 `
  --output results/candidates/shrec2022_test30.json

& .\tools\run_python.ps1 tools/run_cylinder_voting_probe.py `
  --root samples/shrec2022_test30 `
  --output results/candidates/shrec2022_voting.json `
  --max-points 3000 --directions 96 --hypotheses 128 --minimum-support 50
```

`results/` 为可再生输出且受 `.gitignore` 管理；报告记录冻结汇总，原始 JSON/CSV 保留在
本地工作区。
