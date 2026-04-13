# DeepDenoiser MCU Deployment Status

## 1. 项目目标

当前项目目标是将 DeepDenoiser 处理为适合 MCU / STM32 部署的全整数量化模型，并评估其在 STM32F407ZGT6 / X-CUBE-AI 上的可部署性。

当前阶段结论是：现有模型原样直接部署到 STM32F407ZGT6 不现实，需要进一步缩小模型并提升量化友好性。

---

## 2. 当前基线结论

### 2.1 Float 导出链路
已验证原始 TF checkpoint 与 float builtin-only TFLite 严格对齐，结论如下：

- model_input 一致
- 原始 TF vs float builtin-only 的 preds 基本逐点一致
- 原始 TF vs float logits builtin-only：
  - logits mae ≈ 2.64e-06
  - logits rmse ≈ 3.40e-06
  - logits max_abs ≈ 1.62e-05
  - logits argmax_acc = 1.0
  - softmax(logits) mae ≈ 8.42e-08
  - softmax(logits) rmse ≈ 2.56e-07
  - softmax(logits) argmax_acc = 1.0

结论：  
**float builtin-only / float logits builtin-only 导出正确，问题不在 float 导出。**

### 2.2 INT8 PTQ 结果
当前已有 builtin-only 全整数模型：

- `deepdenoiser_int8_builtin.tflite`
- `deepdenoiser_int8_builtin_fullcalib.tflite`

模型格式正确：
- 输入 int8
- 输出 int8
- 中间激活 int8
- bias int32
- 无 Flex / SELECT_TF_OPS / CUSTOM

但精度下降明显，full-calib 版本关键指标如下：

- logits mae ≈ 3.8618
- logits rmse ≈ 4.4226
- logits max_abs ≈ 8.6070
- logits argmax_acc ≈ 0.3449
- softmax(logits) mae ≈ 0.5557
- softmax(logits) rmse ≈ 0.6551
- softmax(logits) argmax_acc ≈ 0.3449

扩大 calibration（从 50 原始文件到 99 原始文件 / 297 个特征）后无明显改善。

结论：  
**当前模型对 PTQ 全整数量化敏感，问题不在 calibration 数量不足。**

### 2.3 ST / X-CUBE-AI 侧结论
ST Edge AI / X-CUBE-AI HOST C-model 验证结果表明：

- ST 生成的 C-model 与 INT8 TFLite 模型高度一致
- cross mae 很小
- cross rmse 很小
- cosine 接近 1

结论：  
**ST 工具链未引入主要误差，当前主要误差来自 INT8 TFLite 相对原始 TF 的量化损失。**

### 2.4 部署侧结论
当前模型参数量、权重大小和激活 RAM 对 STM32F407ZGT6 压力较大，原样直接部署 F407 基本不现实。

---

## 3. 当前工作判断

当前最重要判断如下：

- float 导出没有问题
- INT8 PTQ 当前不可接受
- 扩大 calibration 无明显收益
- 当前模型过大，不适合直接部署 F407

因此，下一步不再继续纠结 float 导图，而应进入以下方向之一：

1. 重新训练一个更适合 PTQ 全整数量化的新浮点模型
2. 或在后续考虑 QAT

当前优先方向为：  
**先训练一个更小、更量化友好的新浮点模型。**

---

## 4. 当前优先实验路线

### 主线优先级
1. 小模型结构设计
2. 小模型 + PTQ
3. 小模型 + 蒸馏 + PTQ
4. 小模型 + QAT（如前述路线仍不足）

### 当前不作为主线
- 剪枝
- 聚类
- 大量组合穷举实验

原因：当前首要矛盾是模型尺寸和 PTQ 敏感性，而不是压缩技术覆盖面不够。

---

## 5. 接下来 3 周执行计划

### 第 1 周：整理基线，不做算法扩展
目标：
- 固定当前可复现实验状态
- 完成版本基线冻结
- 清理仓库管理方式

任务：
- 给当前状态打 tag
- 建立 `archive/` 和 `exp/` 分支
- 补充本状态文档
- 更新 `.gitignore`
- 后续实验统一在实验分支进行

结束标准：
- 明确当前基线版本
- `main` 不继续堆实验产物
- 新实验在独立分支开展

### 第 2 周：只做“小模型 + PTQ”
目标：
- 找到一个比当前模型更小、且 PTQ 后不明显崩溃的候选模型

建议实验：
- A: `depth=4, filters_root=6`
- B: `depth=4, filters_root=4`
- C: `depth=5, filters_root=6, max_filters=48`

统一流程：
- 训练 float checkpoint
- 导出 TF 原始 logits/preds
- 导出 float builtin-only
- 比较 TF vs float logits
- 导出 int8 PTQ
- 比较 TF vs int8 logits
- 用现有评估脚本做候选排序

优先判断顺序：
1. float logits
2. float preds
3. int8 logits
4. 最终波形

建议门槛：
- `int8 softmax argmax_acc < 0.75`：淘汰
- `0.75 ~ 0.90`：可继续优化
- `>= 0.90`：值得继续

结束标准：
- 至少筛出一个更小且 PTQ 表现明显优于当前基线的候选模型

### 第 3 周：只在需要时引入蒸馏
进入条件：
- 小模型 float 精度掉得明显
或
- 小模型 float 可接受，但 PTQ 后仍明显退化

目标：
- 用 teacher-student 蒸馏提高小模型可用性
- 再次评估蒸馏后 PTQ 效果

teacher：
- 当前已验证正确的 float 基线模型

student：
- 第 2 周筛选出的最佳小模型

结束标准：
- 得到一个更小、float 更稳、PTQ 后表现更好的 student 模型

---

## 6. 当前执行原则

- 不再回到“float 导图是否正确”的旧问题
- 不用最终波形代替 logits 判断
- 优先分析：
  - float logits
  - float preds
  - int8 logits
  - 最终波形
- 先解决“模型太大 + PTQ 太敏感”
- 先缩模型，再考虑蒸馏/QAT
- 不做无节制的组合实验扩张

---

## 7. 当前建议结论

当前最推荐路线为：

**先缩小模型结构，再做 PTQ；如果精度不够，再上蒸馏；最后再考虑 QAT。**

不建议当前阶段优先进行：
- 剪枝
- 聚类
- 全组合穷举对比

因为它们会显著增加实验复杂度，但不一定优先解决当前的主要工程问题。

---

## 8. 下一步动作

立即执行：

1. 冻结当前基线版本
2. 建立缩模实验分支
3. 在实验分支中开展小模型训练与 PTQ 评估
4. 使用现有脚本体系复用导出与比较流程
5. 基于 int8 logits 指标筛选候选模型