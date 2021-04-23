# AI2651 Project 1 Part 1
## 如何运行
- 在开始之前：
  - 请保证本项目中的所有文件相对位置正确
  - 请确保data和wavs两个文件夹在当前目录下（或者在代码中重新配置路径）
  - 请确保当前目录下有 `time_domain_features.csv` 和 `freq_domain_features.csv` 两个csv文件（如果没有，请先运行[`short_time_analysis.py`](./short_time_analysis.py))。这是因为分类器的阈值并没有硬编码在代码中，而是需要根据上述两个csv文件确定阈值。
- 要验证特征提取流程和模型在开发集上的表现，请运行[`run_on_def_set.py`](./run_on_dev_set.py)。该文件包括了特征提取、数据分析、构建分类器以及在开发集上验证分类器的全部流程。
- 要使用分类器对测试集上的语音文件进行分类，请运行`run_on_test_set.py`。

## 其他文件
- classifiers文件夹下存放了分类器的实现代码。目前只有[阈值分类器](./classifiers/basic.py)。
- [`short_time_features.py`](./short_time_features.py)存放了提取各项特征的函数。
- [`short_time_analysis.py`](./short_time_analysis.py)提取了开发集上500条音频文件的时域和频域特征，并统计了Voiced和Unvoiced两类帧的特征。分析的结果会被导出到两个.csv文件中，用于构建分类器。
- [`classification.py`](./classification.py)保存了分类任务需要使用的一些工具函数。
- [`parameter_tune.py`](./parameter_tune.py)是报告中随机参数调整的代码。由于实现时疏忽，忘记设定固定的随机数种子，每次运行的结果可能有差异。另外请注意完整运行这部分代码需要很长时间。