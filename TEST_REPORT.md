# PCL 三维几何基元分割拟合：BAGSFit 验证测试报告（阶段一）

日期：2026-09-24　|　状态：BAGSFit 端到端完成（10 样本 × 4 消融配置），SPFN 数据阻塞（见 §7）

## 1. 测试对象与基础设施

- 被测程序：`primitive_fit.exe`（C++17 / PCL / vcpkg 构建，含本次新增的消融开关、失败记录、JSON 输出与两处算法改进）。
- 评测工程：`D:\download\pcl_eval\`，`tools/`（adapt_bagsfit.py、adapt_spfn.py、evaluate.py、batch_test.py）、`samples/`（统一格式）、`results/`（全部输出）。
- 统一样本格式：points.txt + gt_labels.csv（point_index,instance_id,primitive_type）+ gt_primitives.json + metadata.json，点序保持与原始数据一致，非目标类别（cone、无类别支撑面）保留标记。

## 2. 数据集核验结论（实测，非推断）

**BAGSFit TEST-20s**（Google Drive 官方源下载，1.57GB，20 场景 × 36 帧 = 720 帧深度图）：
- npz 为 480×640 深度图：data（带噪）/scan（无噪）/ins（实例，-1=背景）/cls（类别，-1=背景）。
- cls 映射经逐实例几何验证：**0=plane, 1=sphere, 2=cylinder, 3=cone**。
- prim 文件行号 == 实例 ID（平面 6 参数/球 4/圆柱 7/圆锥 7），全部 20 场景交叉验证。
- **源数据 bug（重要）**：prim 中球心 y 坐标符号系统性取反。以无噪 scan 点做代数 LSQ 拟合对照，36/36 球实例在 y 翻转后距离从 0.7–1.9 m 降至 <0.005 m（半径/球心 xz 完全吻合，残差 ~4e-5）。适配器已修正并在 metadata 记录。
- 实例 20/21 为无类别、无参数的场景支撑面（非目标曲面），点云保留、不参与参数误差评价。
- 噪声水平（data−scan）：median 0.0104 m，p90 0.040 m。每帧有效点 14–28 万。

**SPFN TraceParts**：官方服务器 shapenet.cs.stanford.edu 已不可达（DNS 正常但 TLS 无响应；代理/直连/HTTP 均失败）。GitHub 2023 年（SPFN #17）与 2025 年（CPFN #7）issue 均有人求助、无人提供镜像；Wayback Machine 与 HuggingFace 无存档。**数据未能获取，SPFN 测试未执行**，不得视为已完成。适配器（h5 → 统一格式）已按官方加载代码写好并通过代码级核验（noisy_points/gt_labels/{name}_soup_{i} pickle meta，类型取自 meta 字符串而非数字 ID），数据到位即可运行。

样本抽查表（节选，BAGSFit，共核验 10 帧 + 全场景统计）：

| 样本 | 有效点数 | GT 实例 | 类型构成 | 参数 GT | 坐标范围 (m) |
|---|---|---|---|---|---|
| 000_00000 | 215898 | 15 | plane 6, sphere 2, cylinder 3, cone 2, none 2 | 13/15（20/21 无） | x[-0.3,3.3] y[-2.0,1.6] z[0,4.8] |
| 001_00000 | 274394 | 13 | plane 4, sphere 1, cylinder 3, cone 3, none 2 | 11/13 | 约 7×4×5 |
| 006_00018 | 161102 | 15 | plane 5, sphere 2, cylinder 3, cone 3, none 2 | 13/15 | 约 7×4×5 |

标签与点数经程序断言一一对应（适配器内置 assert）。

## 3. 消融实验（10 样本 × A/B/C/D，同阈值同评价口径）

统一参数：distance=0.02 m, minPoints=80, smoothness=12°, curvature=0.05, minInlierRatio=0.3, normalK=300, mergeDistance=0.1, mergeAngle=5°, maxRadius=10。参数在样本 000_00000 上调出（验证集），独立组为其余 9 样本。

独立 9 样本均值（det=IoU≥0.5 检出率，typed=类型一致检出率，ptacc=逐点类型准确率）：

| 配置 | det | typed | mIoU | ptacc | plane F1 | sphere F1 | cyl F1 | 未分配比例 | 平均耗时/帧 |
|---|---|---|---|---|---|---|---|---|---|
| A 全局迭代 RANSAC | 0.13 | 0.16 | 0.270 | 0.63 | 0.66 | 0.13 | 0.06 | 0.00 | 29 s |
| B 生长+独立拟合 | 0.47 | 0.42 | 0.481 | 0.75 | 0.77 | 0.39 | 0.69 | 0.26 | 9 s |
| C B+合并 | 0.56 | 0.55 | 0.547 | 0.75 | 0.77 | 0.39 | 0.69 | 0.26 | 9 s |
| D C+重拟合+重吸收 | 0.51 | 0.54 | 0.549 | 0.81 | 0.76 | 0.37 | 0.71 | 0.13 | 9 s |

结论：区域生长相对全局 RANSAC 是决定性的（det 0.13→0.47，mIoU 0.27→0.48，且 A 反而最慢）；区域合并进一步显著提升（det 0.47→0.56）；合并后重拟合+重吸收把未分配点从 0.26 降到 0.13、逐点类型准确率 0.75→0.81，mIoU 持平，但检出率略降（0.56→0.51，重吸收在边界点归属上的双刃剑效应，见 §5）。

## 4. 几何参数误差（D 配置，匹配且类型一致，法向/轴向符号已对齐）

| 基元 | 指标 | n | median | p90 | max |
|---|---|---|---|---|---|
| plane | 法向角误差 (°) | 26 | 0.60 | 1.88 | 4.05 |
| plane | 偏移距离 (m) | 26 | 0.011 | 0.034 | 0.11 |
| sphere | 球心距离 (m) | 15 | 0.007 | 0.020 | 0.13 |
| sphere | 半径相对误差 | 15 | 2.7% | 5.6% | 57.2% |
| cylinder | 轴向角误差 (°) | 17 | 0.65 | 2.15 | 4.02 |
| cylinder | 轴线距离 (m) | 17 | 0.002 | 0.010 | 0.026 |
| cylinder | 半径相对误差 | 17 | 1.8% | 7.3% | 12.1% |

中位误差与噪声水平一致（偏移 0.011 m ≈ 噪声中位 0.0104 m）。球面 max 57% 半径误差为可见曲面过小导致的参数不稳定样本（已按任务书单独记录，见 evaluation.json 的 parameter_errors）。

## 5. 失败模式与针对性改进（保留改进前后指标）

失败案例：
- **大平面过分割**：支撑大平面（5 万点级）被区域生长切成数百至数千碎块（如 000 场景 652 块），合并仅部分恢复（该实例 IoU 0.17–0.37）。法向噪声使生长断裂是根因。
- **圆柱侧壁误判为球/平面**：碎片化后小块曲面被平面或大半径球捕获（圆柱 GT 点被预测为 sphere 的比例最高约 5%）。
- **非目标曲面被强行拟合**：cone 点 79% 被分配（多被拟合成 cylinder——锥面与大端半径圆柱局部不可分，属方法边界）；无类别支撑面 60% 被拟合成 plane/sphere（几何上合理，但无参数 GT，评价中如实计入"非目标被拟合"）。

算法改进（本轮实施，D 配置）：
1. 贪心合并迭代至不动点（单轮合并会漏掉吸收后才相容的配对）。
2. 合并后全局重吸收 + 法向一致性约束（吸收条件：距离 ≤ tolerance 且点法向与曲面法向夹角 ≤30°）。

改进前后（调参样本 000_00000，D 配置）：det 0.40→0.53，typed 0.45→0.64，mIoU 0.479→0.589，ptacc 0.76→0.82，未分配 0.26→0.11。无约束重吸收曾在 002_00000 上把 det 从 0.43 拉到 0.21（边界点稀释 IoU），加法向约束后回升至该样本 C 配置水平，但仍比 C 低——重吸收策略需要进一步细化，这是下一步工作。

## 6. 参数敏感性（6 轮粗扫，调参样本）

- **normalK**：最敏感。30 → 300 使 det 0→0.40（稠密深度图上小邻域法向被噪声淹没）。
- **minInlierRatio**：0.8→0.3 使 det 0.27→0.40、ptacc 0.52→0.74（区域跨边界时 80% 内点比例过苛）。
- **curvature**：0.1→0.05 使 typed 0.45→0.64（更早停止生长，减少跨曲面泄漏）。
- distance 0.02/0.03、mergeDistance 0.06/0.1、smoothness 8–15° 均为二阶影响；maxRadius 1.0 反而轻微变差。

## 7. 遗留问题与下一步

1. SPFN 数据阻塞：需官方服务器恢复或持有数据者提供（来源必须可追溯）；适配器已就绪。
2. 大平面合并失败：考虑基于点数的自适应合并阈值或图聚类式合并替代贪心。
3. 重吸收与检出率的权衡：按"吸收后必须保持与模型法向场连续"再收紧。
4. 扩大规模：当前 10/720 帧（5 场景 × 2 帧），后续可全量跑 720 帧（预计 D 配置约 1.8 小时）。
5. 拟合失败/崩溃统计：40 次运行 0 崩溃、0 NaN/Inf 参数（evaluation.json failures 字段记录）。

## 附：交付物索引（D:\download\pcl_eval\）

- `tools/adapt_bagsfit.py`、`tools/adapt_spfn.py`：数据适配器（统一格式，含修正记录）
- `tools/verify_bagsfit.py`：标签/参数几何验证脚本
- `tools/evaluate.py`：统一评价（类型/实例匈牙利匹配/合并/参数误差/失败统计）
- `tools/batch_test.py`：批量测试（A/B/C/D，自动评价）
- `results/bagsfit_ablation.csv`、`results/bagsfit_summary.csv`：指标汇总
- `results/bagsfit/<sample>/<mode>/`：每样本每模式的 labels/models/primitives JSON/彩色 PCD/评价 JSON
- `samples/bagsfit/`：统一格式样本（GT 已修正球心 y 符号并在 metadata 注明）
- 被测程序源码：`D:\download\pcl_region_primitive\pcl_region_primitive\main.cpp`
