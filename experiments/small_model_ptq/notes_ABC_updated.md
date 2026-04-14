# Small-Model PTQ Notes (A / B / C)

## Experiment Purpose

本轮实验的目标不是直接把原始 DeepDenoiser 上 MCU，而是先找到一个：

1. 结构更小
2. PTQ 友好
3. 对 original_float 不至于退化过大

的小模型主线。

已完成三条结构：
- A_d4_r6
- B_d4_r4
- C_d5_r6_cap48

---

## Final High-Level Conclusion

最终结论很明确：

- `A_d4_r6`：当前主线
- `B_d4_r4`：极限压缩备选
- `C_d5_r6_cap48`：当前不继续深挖

原因不是单看某一个指标，而是综合：

- waveform 对 original_float 的接近程度
- fixed-sample PTQ 稳定性
- 模型大小 / MCU 友好性

A_d4_r6 在这三者之间平衡最好。

---

## A / B / C What They Mean

### A_d4_r6
- depth=4
- filters_root=6

这是第一条成功跑通的“小模型 PTQ baseline”。
它不是 original 模型的近似复制品，但已经证明：
- float 导出正确
- int8 PTQ 很稳
- waveform 在三者里最接近 original_float

因此 A 是当前最佳主线。

### B_d4_r4
- depth=4
- filters_root=4

B 的核心价值是更小。
它证明了：
- 结构还能继续压
- 但压到这个程度后，PTQ 稳定性和 waveform 相似度都比 A 更差

所以 B 更适合做“极限压缩备选”，而不是当前主线。

### C_d5_r6_cap48
- depth=5
- filters_root=6
- filters_cap=48

C 的核心思想是：
- 更深
- 但用 cap48 限制通道暴涨

它的结果说明：
- float 导出正确
- int8 也很稳
- 但体积显著更大
- waveform 并没有赢过 A

所以 C 现在不值得继续深挖。

---

## What Was Learned

### 1. Float export is no longer the problem
A / B / C 三条线都已经证明：
- TF logits vs float logits 基本严格对齐
- argmax_acc 都是 1.0

因此后续精力不该再浪费在“float 导图是不是错了”上。

### 2. PTQ sensitivity depends strongly on structure
- A：稳
- B：更小，但退化更明显
- C：也稳，但明显更大

说明“量化友好”确实和结构容量、深度、通道分配强相关。

### 3. A is the best balance point so far
A 不是最小，也不是最深，但它在：
- original 相似度
- PTQ 稳定性
- 体积

三者之间最平衡。

---

## Why Not Continue More Architecture Search Now

不建议现在继续扩更多结构点，比如再试一堆：
- d4_r5
- d5_r5_cap40
- d4_r6_cap32
- ...

原因：
1. 现在已经有足够证据表明 A 是当前最优平衡点
2. 继续扩结构会让实验空间迅速膨胀
3. 当前真正短板不是“还没找到结构”，而是：
   - A 还没尽量贴近 original_float

所以更合理的下一步不是继续搜结构，而是：
- 固定 A
- 提升 A

---

## Next Step: Distill A

下一阶段建议：

### Teacher
- original large float model

### Student
- A_d4_r6

### Goal
让 A_d4_r6 在不显著增大模型的前提下，更接近 teacher / original_float。

### Why Distillation
因为 A 当前最大问题不是量化崩，而是：
- 虽然量化友好
- 但和 original_float 还有明显差距

蒸馏正好适合解决这个问题。

---

## Distillation Strategy (v1)

推荐先做最小改动版：

### Keep structure fixed
- student 结构仍然是 A_d4_r6

### Keep original supervised loss
- 继续保留现有 cross_entropy

### Add teacher guidance
- 用 teacher logits / softmax(logits) 作为蒸馏目标

### First version
- total_loss = alpha * hard_label_ce + beta * distill_loss
- 第一版可从：
  - alpha = 0.5
  - beta = 0.5
  - temperature = 2 or 3
开始

---

## What To Avoid Right Now

当前不建议直接优先做：

### pruning
因为现在主要问题不是再压一丁点体积，而是效果保持。

### clustering
对当前主问题帮助不如蒸馏直接。

### full QAT immediately
QAT 有价值，但应排在：
- 先蒸馏
- 再 PTQ
之后。

---

## Recommended Roadmap

### Stage 1
固定 A_d4_r6 为主线结构

### Stage 2
做 `A_d4_r6_distill_v1`

### Stage 3
对蒸馏后的 A 重复当前评估链：
1. TF raw debug
2. float builtin-only
3. TF vs float logits
4. int8 builtin-only
5. TF vs int8 logits
6. 30-sample waveform compare vs original_float

### Stage 4
如果蒸馏后的 A 仍然在 PTQ 上不够理想，再考虑 QAT

---

## Final Note

到目前为止，A / B / C 已经足够回答“当前哪个结构最值得继续”。

答案不是：
- B，因为更小
- 也不是 C，因为更深

而是：

> A_d4_r6 是当前最值得继续做蒸馏和后续部署验证的结构。
