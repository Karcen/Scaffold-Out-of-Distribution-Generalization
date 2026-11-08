# ADMET frontier 修订代码说明

本文代码对应 *Model-dependent ADMET performance under descriptor and fingerprint frontier evaluation*。本目录是原仓库的日期化增补；旧研究文件与已有 MIT 许可证保持不变。完整命令见 [英文说明](README.md)。

## 研究问题

描述符 frontier 与 Morgan frontier 如何改变六项 ADMET 任务的测试分子、预测表现和模型排序？两种几何共用分组与分配规则，但会选中不同分子、类别比例和回归标签范围。

## 结果

本次公开的是当前科学代码、实验与统计方案、环境配置、执行源码快照及部分聚合结果。包含传统基线、Morgan MLP、图模型基础组件、Chemprop、随机与反向分配、DILI 联合排序不确定性、结构支持和敏感性分析。未加入任何新实验结论。已附聚合结果可支持七张图的重绘；其余三张图依赖未包含的分子或分组诊断记录。

本目录不包含稿件、原始论文档案、原始分子数据、逐分子标签与预测、模型权重、私人模板或管理文件。不要将此代码发布表述为完整结果档案已经公开。

## 意义

代码明确区分模型损失、测试集组成和评估规则。不同 frontier 的差异包含测试和验证人群的变化；不能直接解释为固定测试集上几何选择的因果效果。Bootstrap 区间以已拟合模型及实际测试分组为条件，多个拟合 seed 不代表独立化学域重复。

## 方法

从本日期目录运行命令。使用 `requirements-core.txt` 和批次环境记录建立通用环境；Chemprop 使用独立环境和其依赖冻结文件。TDC 数据获取脚本会记录输入哈希；只有显式传入 `--download` 才下载缺少的数据。原仓库的 `results/splits` 用于历史划分一致性核对。

```sh
python codes/prepare_public_inputs.py --download --reference-root ../..
```

拟合命令会耗时，并生成各自的配置、split manifest 和逐分子预测。本次发布准备没有运行这些实验。不要修改代码或方案后恢复先前冻结的运行。

新版作图需要外部 `nature-figure` 质量核验工具，使用环境变量指定本机安装位置：

```sh
export NATURE_FIGURE_SKILL=/path/to/nature-figure
python codes/plot_available_figures.py
```

PDF/SVG 保留可编辑文字，另导出 PNG。单纯代码公开不改变第三方数据许可；原始 TDC 数据按原来源获取和使用。已有仓库 MIT 许可证不覆盖第三方数据或依赖包。
