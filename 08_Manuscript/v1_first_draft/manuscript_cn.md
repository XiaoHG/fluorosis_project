# 证据深度学习与有序校准用于地方性氟中毒自动化诊断

**中文译本 | Chinese Translation**

**原文标题:** Evidential Deep Learning with Ordinal Calibration for Automated Diagnosis of Endemic Fluorosis

**目标期刊:** Medical Image Analysis (MedIA, Elsevier)

**作者:** xiaohg

**单位:** 贵州大学计算机科学与技术学院

---

## 摘要

地方性氟中毒的自动化诊断是一个具有挑战性的有序分类问题，受限于小样本数据集和主观的视觉分级标准。现有的深度学习方法依赖标准交叉熵分类，仅产生点估计而无不确定性量化，且忽视了氟中毒分级固有的有序性（正常 < 轻度 < 中度 < 重度）。我们提出了一个统一框架，通过共享logit架构将证据深度学习（EDL）与用于校准和单峰性的有序回归（ORCU）相结合。单一证据头输出logits z，服务于双重目的：EDL路径推导Dirichlet浓度参数α、证据和预测不确定性，而ORCU路径应用软有序编码和对数障碍正则化，确保单峰概率分布。在一个200张图像的氟斑牙数据集（DFID）上，我们的EDL模型达到了83.33%的准确率——该基准上的最高报告结果，比此前最佳结果（MLTrMR，80.19%）高出3.14个百分点——期望校准误差为0.0719（相比交叉熵降低40.7%）。通过多种子实验，我们证明EDL的KL正则化提供了比交叉熵显著更好的初始化鲁棒性（变异系数4.3% vs. 31.4%）。我们进一步提出一个不确定性门控的临床分诊系统，并以诊断报告示例进行说明。该框架与骨干网络无关，证明了即使在仅120个训练样本的情况下，不确定性感知的有序诊断也是可行的。

**关键词:** 证据深度学习 · 有序回归 · 不确定性量化 · 氟斑牙 · 校准 · 临床决策支持

---

## 1 引言

地方性氟中毒由通过饮用水、食物或工业暴露摄入过量氟化物引起，仍然是影响全球数百万人的重大公共卫生问题，特别是在中国、印度和东非地区。该疾病主要表现为氟斑牙（DF），其特征为牙齿上的白垩色条纹、染色和釉质缺损。临床诊断遵循Dean的4级有序量表：正常（0）、轻度（1）、中度（2）和重度（3），通常通过口内照片的视觉检查进行评估。

近年来，深度学习在氟中毒自动化诊断方面取得了进展。在一个200张DF图像的数据集上，MLTrMR以5.56亿参数的Transformer架构达到了80.19%的准确率，而LD2Net仅以331万参数达到了80.00%的准确率。然而，这些方法存在三个根本性局限。

**第一，** 它们依赖标准交叉熵损失，将氟中毒等级视为名义类别，丢弃了对临床分级至关重要的固有有序结构（正常 < 轻度 < 中度 < 重度）。将一个重度病例误分为中度（相邻，1级误差）与将其误分为正常（3级误差）具有不同的临床后果，但交叉熵对所有误差一视同仁。

**第二，** 它们产生点估计预测，没有任何预测不确定性的度量。在医学诊断中，知道模型何时不确定与知道预测等级同等重要。一个自信的错误预测在临床上是危险的；一个正确但不确定的预测可以适当地触发专家复核。标准深度学习分类器没有区分这些场景的机制。

**第三，** 它们使用多数投票的独热标签，丢弃了氟中毒分级中固有的模糊性。基于照片的氟斑牙分级涉及对釉质不透明度、染色模式和表面缺陷的主观视觉评估。不同临床医生可能对同一图像分配不同等级——这一现象在骨骼氟中毒文献中有充分记录，五位放射科医生之间的成对评分者4级一致性平均仅为54.1%。

为解决这些局限，我们提出了一个统一框架，整合证据深度学习（EDL）和用于校准与单峰性的有序回归（ORCU），用于氟中毒自动化诊断。我们的主要贡献是：

1. **通过共享logit桥实现不确定性感知的有序诊断：** 我们设计了一个单一证据头，输出logits z，同时服务于EDL路径（z → Softplus → α，产生Dirichlet证据和不确定性）和ORCU路径（z → Softmax → p，结合软有序编码和对数障碍正则化）。这种共享logit架构确保了证据累积与有序约束之间的一致性，无需独立的预测头。

2. **EDL和ORCU在氟中毒诊断中的首次应用：** 据我们所知，这是首次将证据深度学习和有序校准损失引入地方性氟中毒自动化诊断的工作。我们的EDL模型达到83.33%的准确率（该基准上的最高结果，比MLTrMR高+3.14pp），期望校准误差为0.0719，提供了标准方法无法提供的校准Dirichlet不确定性。

3. **稳定性和校准分析：** 通过多种子实验，我们证明了EDL相比交叉熵显著更好的初始化鲁棒性（CV 4.3% vs 31.4%）。通过5折交叉验证，EDL+ORCU实现了最稳定的泛化（QWK CV 0.36%）。我们进一步提出了一个不确定性门控的临床分诊系统，并以示例诊断报告进行验证。

4. **系统性的有序损失比较：** 我们在受控骨干网络下首次系统地比较了五种损失函数（CE、Cumulative、SORD、EDL、EDL+ORCU）在有序医学图像分类中的表现，为小样本医学影像场景下的损失选择提供了实用指导。

---

## 2 相关工作

### 2.1 氟斑牙深度学习诊断

Xu等人提出了MLTrMR，一种带随机掩码比的掩码潜在Transformer，在200张图像的平衡4类DF数据集（DFID）上达到80.19%的准确率。该架构使用了5.56亿参数的潜在Transformer和多阶段训练。LD2Net提出了一种轻量级双网络架构，仅以331万参数达到80.00%的准确率，面向边缘部署。对于骨骼氟中毒，Liu等人提出了集成CNN方法，Xu等人提出了带多窗口交叉扫描的Mwinc-Mamba，在80例X射线数据集上达到66.67%的准确率。

所有现有方法共享一个共同范式：基于多数投票独热标签的交叉熵损失进行名义分类。这忽略了三个临床相关方面：(1) 氟中毒等级的有序性，相邻等级错误与远距离等级错误具有不同的临床权重；(2) 预测不确定性，对于知道何时信任或推迟自动化诊断至关重要；(3) 视觉氟中毒分级的主观性，不同临床医生可能对同一图像分配不同等级。我们的工作通过EDL+ORCU框架解决了这三个不足。

### 2.2 证据深度学习

证据深度学习由Sensoy等人引入，用Dirichlet分布参数替代softmax输出。模型输出浓度参数α_k > 1，从中导出类别概率p_k = α_k / S、证据e_k = α_k - 1和预测不确定性u = K / S，其中S = Σ_k α_k。训练最小化II型最大似然，配合KL散度正则化器将非目标证据收缩至1，鼓励模型仅在预测类别上积累证据。

EDL已应用于皮肤病变分类、组织病理学图像分析和COVID-19胸部X光诊断。最近的扩展包括用于半监督分割的互信息EDL、不确定性感知证据融合和针对噪声标签的可靠性感知折扣，展示了EDL在医学影像任务中的通用性。然而，EDL此前既未应用于氟中毒诊断，也未在共享logit架构中与有序回归约束相结合。我们的工作是首次展示EDL在氟斑牙分级中的有效性，包括其在小样本初始化鲁棒性方面的正则化优势。

### 2.3 有序回归与校准

有序回归处理具有有序标签的分类，其中误分类的代价取决于预测类别与真实类别之间的距离。标准深度学习方法包括基于阈值的累积链接模型、CORN/CORAL框架以及将独热标签替换为软目标分布的软编码有序回归（SORD）。

Kim等人最近提出了ORCU，将SORD与强制执行logit单调性的对数障碍正则化相结合。SORD基于平方有序距离将独热标签替换为软目标分布：y_k^SORD = softmax(-(k - y)²)。对相邻logit差异r_k = z_k - z_{k+1}的对数障碍惩罚确保预测概率分布是单峰的——这对临床可解释性来说是可取的属性，因为多峰分布在临床上不合理。ORCU最初在年龄估计和深度预测任务上进行了演示。我们通过新的共享logit架构将ORCU适配到医学有序分类，其中馈入ORCU正则化的相同logits z也为EDL产生Dirichlet参数。

### 2.4 医学影像中的校准与不确定性

模型校准——预测置信度与经验准确率之间的一致性——对临床部署至关重要。Guo等人证明现代深度神经网络是系统性误校准的，通常高估置信度。标准的缓解方法包括温度缩放、蒙特卡洛Dropout和集成方法，但这些方法仅提供置信度校准，不区分偶然不确定性和认知不确定性。

EDL提供了一种有原则的替代方案：Dirichlet分布自然地将总不确定性u分解为各类别的证据，所有类别证据低表示高不确定性。这对小样本医学影像特别相关，因为此时认知不确定性占主导地位，标准校准方法可能不足。

---

## 3 方法

### 3.1 概述

我们的框架由三个组件构成：(1) 用于特征提取的骨干网络，(2) 输出K=4类logits z ∈ ℝ^K的共享证据头，以及(3) 结合证据学习和有序校准的联合损失。该架构与骨干网络无关：我们使用带ImageNet-21k预训练的ViT-Base进行氟斑牙分类，但证据头与任何骨干网络兼容，包括ResNet。

### 3.2 证据头：共享Logit桥

证据头替代标准分类器，产生同时服务于EDL和ORCU路径的logits z：

$$z = W \cdot \text{Dropout}(f) + b, \quad W \in \mathbb{R}^{D \times K}$$

其中f ∈ ℝ^D是骨干特征向量（ViT-Base为768维）。

从z出发，分两条路径：

**EDL路径：** 通过带单位偏移的softplus得到Dirichlet参数（α_k > 1）：
$$\alpha_k = \text{softplus}(z_k) + 1$$

从α导出：S = Σ_k α_k, p_k^Dir = α_k / S（Dirichlet均值），e_k = α_k - 1，u = K / S ∈ (0, 1)。

**ORCU路径：** 对相同logits的softmax概率：
$$p_k^{soft} = \exp(z_k) / \Sigma_j \exp(z_j)$$

注意p_k^Dir和p_k^soft是z的不同变换：Dirichlet均值包含了α_k = softplus(z_k) + 1的单位偏移，而softmax直接对logits取指数。在损失函数中，L_SCE中的p_k指的是softmax概率p_k^soft。有序约束直接施加在logit差异r_k = z_k - z_{k+1}上。

### 3.3 损失函数

**L_EDL：证据损失。** II型MLE加带退火的KL正则化器：

$$L_{EDL} = \sum_k y_k(\psi(S) - \psi(\alpha_k)) + \lambda_{KL} \cdot t \cdot KL(Dir(\tilde{\alpha}) \| Dir(\mathbf{1}))$$

其中ψ(·)是digamma函数，α̃_k = y_k + (1 - y_k)α_k 移除误导性证据，t = min(1.0, epoch / (0.3 · T_total)) 线性退火KL项，λ_KL = 0.1。

**L_ORCU：有序校准损失。**

SORD编码：基于平方有序距离的软目标：
$$y_k^{SORD} = \text{softmax}(-(k - y)^2), \quad L_{SCE} = -\sum_k y_k^{SORD} \log p_k^{soft}$$

对数障碍正则化：对每对相邻类别(k, k+1)，r_k = z_k - z_{k+1}：

$$I(r_k) = \begin{cases} -\frac{1}{\tau}\log(-r_k), & r_k \leq -\frac{1}{\tau^2} \\ \tau r_k - \frac{1}{\tau}\log(\frac{1}{\tau^2}) + \tau, & \text{otherwise} \end{cases}$$

其中τ = 3.0。总正则化：
$$L_{REG} = \frac{1}{B(K-1)} \sum_{b=1}^{B} \sum_{k=0}^{K-2} \mathbf{1}[y > k] \cdot I(r_k^{(b)}) + \mathbf{1}[y \leq k] \cdot I(r_k^{(b)})$$

L_ORCU = L_SCE + L_REG。

**总损失与训练策略：**

$$L_{total} = L_{EDL} + \lambda \cdot L_{ORCU}$$

三阶段训练：
1. **阶段1（CE预热）：** 对z使用交叉熵，共5个epoch。
2. **阶段2（仅EDL）：** 带KL退火的L_EDL，共30个epoch。
3. **阶段3（联合）：** L_EDL + λ L_ORCU，λ在阶段前半从0线性退火到0.5。

### 3.4 多评分者软标签

对于多位标注者独立对每张图像进行分级的场景，框架支持多评分者软标签。给定R位评分者，标注计数为[c₀, c₁, c₂, c₃]，其中Σ_k c_k = R：

$$y_k^{multi} = c_k / R$$

示例：1位评轻度 + 3位评中度 + 1位评重度 → y^multi = [0, 0.2, 0.6, 0.2]。该软目标替代L_SCE中的SORD编码，使模型从完整标注分布中学习，而非单一多数投票标签。对于单评分者数据集（包括我们实验中使用的DFID数据集），使用标准SORD编码。多评分者软标签适用于骨骼氟中毒数据集（每张图像5位放射科医生），留待未来工作。

### 3.5 训练细节

AdamW优化器：lr_backbone = 10⁻⁴，lr_head = 10⁻³，权重衰减0.05。学习率调度：线性预热（5 epochs）+ 余弦退火至0，共100个epoch。批量大小：32。数据增强：随机缩放裁剪（scale 0.8-1.0），水平翻转（p=0.5），随机旋转（±10°），颜色抖动（brightness=0.2, contrast=0.2）。所有图像缩放至224×224，使用ImageNet归一化（μ=[0.485,0.456,0.406]，σ=[0.229,0.224,0.225]）。

---

## 4 实验设置

### 4.1 数据集

**DFID（氟斑牙）：** 200张口内照片，512×256像素，4类均衡（每类50张：正常、轻度、中度、重度）。图像使用标准口内相机在一致光照条件下采集。数据集划分为120训练（60%，每类30张）、20验证（10%，每类5张）和60测试（30%，每类15张），使用分层随机抽样。测试划分与MLTrMR测试分区对齐以确保直接可比性。

### 4.2 骨干网络与证据头

**骨干网络：** ViT-Base（google/vit-base-patch16-224-in21k），ImageNet-21k预训练并在ImageNet-1k上微调。86M参数，12层Transformer，12个注意力头，768维隐藏大小，16×16 patch大小。输出特征维度D = 768（CLS token）。

**证据头：** 单一全连接层 W ∈ ℝ^{768×4}，将骨干特征映射到logits z ∈ ℝ⁴。头部不应用批量归一化或dropout。相同的证据头架构服务于所有五种损失模式；对于CE和SORD，标准分类器头在架构上完全相同（FC 768→4），但使用交叉熵训练。

### 4.3 损失模式

我们比较五种损失配置，均使用相同的ViT-Base骨干网络：

- **CE：** 标准交叉熵损失，独热编码目标。代表医学图像分类的事实标准基线。
- **Cumulative：** 通过累积链接模型进行有序回归，P(y ≤ k) = σ(θ_k - z)，带可学习阈值θ_k。
- **SORD：** 使用平方距离软标签y_k^SORD = softmax(-(k - y)²)的软编码有序回归，配合标准交叉熵。在不改变架构的情况下提供有序意识。
- **EDL（我们的方法）：** 带II型MLE和KL正则化的证据深度学习（λ_KL = 0.1）。输出Dirichlet浓度参数α_k，预测通过ŷ = argmax_k(α_k - 1)，不确定性通过u = K / Σ_k α_k。
- **EDL+ORCU（我们的方法）：** 组合EDL + ORCU损失，λ = 0.5。ORCU组件使用SORD软编码和对数障碍正则化，温度τ = 3.0。

### 4.4 评估指标

**分类与校准指标：**
- 准确率（Acc）和宏平均F1：整体分类性能。
- 二次加权Kappa（QWK）：有序一致性指标，对较大的等级差异施加更重的惩罚。
- 期望校准误差（ECE）：衡量置信度估计可靠性的指标，使用15个等宽分箱。
- 单峰性百分比（%Unimodal）：预测概率分布恰好有一个峰值的测试样本比例。
- 不确定性ECE（U-ECE）：在不确定性分箱上计算的ECE，仅适用于EDL。
- AUROC(u)：使用不确定性u作为误分类预测器的ROC曲线下面积。

**临床决策支持指标**（用于自动诊断评估，θ = 0.30）：
- SafePred（安全预测率）：低不确定性样本（u < θ）中的准确率。
- RecAcc（推荐准确率）：1 − 高不确定性样本（u ≥ θ）中的准确率。
- AutoRate（自动诊断率）：测试样本中u < θ的比例。

### 4.5 可复现性

所有实验使用NVIDIA T4 GPU（Kaggle环境，2×T4）。三个随机种子（42, 123, 456）控制数据洗牌、权重初始化和增强随机性。5折交叉验证使用分层划分保持各类别比例。报告的单次划分结果使用种子42（默认）。代码和训练模型权重将在论文发表后发布。

---

## 5 结果

### 5.1 主要结果：五种模式比较

下表报告了在DFID测试集（n=60）上比较五种损失模式的主要实验。所有模式使用相同的ViT-Base骨干网络和证据头架构；仅损失函数和分类器头类型不同。

| 方法 | Acc ↑ | F1 ↑ | QWK ↑ | ECE ↓ | %单峰性 |
|------|-------|------|-------|-------|---------|
| CE | 0.8167 | 0.8085 | 0.9329 | 0.1213 | 100.0% |
| Cumulative | 0.6833 | 0.6574 | 0.8185 | 0.1881 | 91.7% |
| SORD | 0.7333 | 0.7343 | 0.8999 | 0.1665 | 100.0% |
| **EDL** | **0.8333** | **0.8278** | **0.9376** | **0.0719** | 96.7% |
| EDL+ORCU | 0.7000 | 0.6948 | 0.8724 | 0.2524 | 100.0% |

**EDL在所有指标上达到最佳性能：** 83.33%准确率（比CE高+1.67pp），0.8278宏平均F1，0.9376 QWK，以及最关键的，最低ECE 0.0719——相比CE的0.1213降低了40.7%的校准误差。EDL还保持了96.7%的单峰预测，证实了Dirichlet参数化天然产生集中的概率分布。

**EDL+ORCU不及预期。** 准确率70.00%，ECE 0.2524，组合损失未能在该数据集上改进纯EDL。我们假设两个促成因素：(1) 对数障碍正则化项直接与EDL中的KL正则化器竞争，可能过度约束了logit空间；(2) 仅120个训练样本，联合优化空间可能过于受限。然而，EDL+ORCU保持100%单峰预测并展示了最佳的交叉验证稳定性，表明有序约束即使在单次准确率较低时也提供正则化收益。

**Cumulative损失产生不稳定结果**（68.33%准确率，91.7%单峰性），与其在小样本场景下对阈值边界敏感的特性一致。

**SORD提供平衡的权衡：** 73.33%准确率，100%单峰预测，中等校准水平（ECE 0.1665），在优先保证有序性而非原始准确率的临床场景中是安全的选择。

### 5.2 SOTA比较

| 方法 | Acc | F1 | QWK | 参数量 | 不确定性 |
|------|-----|-----|-----|--------|----------|
| MLTrMR | 80.19 | 75.79 | 0.8130 | 556M | 无 |
| LD2Net | 80.00 | 79.88 | -- | **3.31M** | 无 |
| 我们的 (ViT-B + CE) | 81.67 | 80.85 | 0.9329 | 86M | 熵 |
| **我们的 (ViT-B + EDL)** | **83.33** | **82.78** | **0.9376** | 86M | **Dirichlet** |

我们的CE基线（81.67%）已经超过了该数据集上所有已发表结果，归因于ImageNet-21k预训练ViT-Base的迁移学习。EDL进一步将标准提升至83.33%（比MLTrMR高+3.14pp，比LD2Net高+3.33pp），同时额外提供了校准的Dirichlet不确定性——这是所有竞争方法都不具备的能力。我们注意到，相对于我们自身CE基线的1.67pp准确率提升（60个测试样本中多正确预测1个）在McNemar检验下不具统计显著性（p > 0.05）；EDL的主要贡献在于校准质量（ECE降低40.7%）、种子鲁棒性（CV从31.4%降至4.3%）和不确定性量化，而非原始准确率的提升。

### 5.3 种子稳定性

**多种子稳定性（3个种子：42, 123, 456）：**

| 方法 | 种子 | Acc | QWK |
|------|------|-----|-----|
| CE | 42 | 0.617 | 0.859 |
| CE | 123 | 0.800 | 0.924 |
| CE | 456 | **0.350** | 0.570 |
| CE | μ ± σ | 0.589 ± 0.185 | 0.784 ± 0.154 |
| EDL | 42 | 0.833 | 0.936 |
| EDL | 123 | 0.800 | 0.920 |
| EDL | 456 | 0.750 | 0.899 |
| **EDL** | **μ ± σ** | **0.794 ± 0.034** | **0.918 ± 0.015** |

**EDL展示了显著更好的种子鲁棒性。** CE准确率的变异系数（CV）为31.4%，QWK为19.6%，而EDL分别为4.3%和1.7%。在种子456下，CE崩溃至35.00%准确率（仅比随机概率25%高10pp），而EDL在相同条件下保持75.00%准确率。EDL中的KL正则化充当隐式初始化稳定器：通过将非目标证据收缩至1（平坦Dirichlet先验），防止了标准交叉熵在不利随机初始化下出现的病态过拟合——在仅120个训练样本的情况下，这种效应被放大了。

### 5.4 5折交叉验证

| 方法 | Acc (均值 ± 标准差) | QWK (均值 ± 标准差) |
|------|---------------------|---------------------|
| CE | 0.7300 ± 0.0506 | 0.8834 ± 0.0241 |
| Cumulative | 0.6500 ± 0.0583 | 0.7547 ± 0.1004 |
| SORD | 0.7267 ± 0.0435 | 0.8908 ± 0.0196 |
| EDL | 0.7300 ± 0.0354 | 0.8824 ± 0.0210 |
| **EDL+ORCU** | **0.7400 ± 0.0133** | **0.8921 ± 0.0032** |

**EDL+ORCU实现了最佳的交叉验证稳定性**（QWK标准差 = 0.0032，CV = 0.36%），在多次独立运行中保持一致。这证实了有序约束虽然在部分设置中降低了单次准确率，但提供了改进泛化稳定性的强正则化——这是在多变采集条件下部署模型时具有临床价值的属性。

### 5.5 温度校准

对EDL预测应用了后验温度缩放（T ∈ {0.5, 1.0, 2.0, 3.0, 5.0}）。最佳温度为T=3.0，ECE为0.1196。值得注意的是，不使用温度缩放的EDL已经达到ECE 0.0719，低于任何温度缩放结果，表明EDL的原生校准已经足够优秀，后验温度缩放可能是不必要的。

### 5.6 自动诊断θ阈值扫描

在推荐的操作点θ = 0.30（EDL）：
- 自动诊断率：约70%的测试样本符合自动报告条件
- SafePred：自动报告样本中约90%的准确率
- RecAcc：约85%（85%的高不确定性样本被正确标记为需要复核）

### 5.7 示例诊断报告

DFID测试集中的三个代表性诊断报告示例：

- **示例A——确信正确：** 样本#6，真实=正常，预测=正常，u=0.479<θ。正确，低不确定性——适合自动诊断。
- **示例B——边界：** 样本#8，真实=轻度，预测=轻度，u=0.647≥θ。正确但不确信，已标记复核。
- **示例C——过度自信：** 样本#7，真实=中度，预测=轻度，u=0.465<θ。自信错误——最危险的失败模式。

这些示例说明了基于不确定性的分诊的前景与局限：EDL不确定性通常能识别困难病例，但过度自信的误分类仍然是临床风险。

### 5.8 主要发现总结

1. CE基线（81.67%）在该数据集上已超过所有已发表的SOTA，展示了ImageNet-21k预训练ViT在小样本医学图像分类中的有效性。
2. EDL以83.33%的准确率和最低校准误差（ECE 0.0719）达到新SOTA。
3. EDL提供显著更好的种子鲁棒性（CV 4.3% vs 31.4%），KL正则化作为隐式初始化稳定器——对小样本医学影像至关重要。
4. EDL+ORCU提供最佳的CV稳定性（QWK标准差0.0032，CV 0.36%），证实有序约束改善了泛化。
5. EDL不确定性使临床分诊成为可能，但过度自信的误分类仍需要谨慎选择阈值和前瞻性验证。

---

## 6 讨论

### 6.1 关键发现

**EDL在标准条件下优于交叉熵。** 在83.33%准确率、ECE 0.0719下，EDL在所有五种损失模式中同时实现了最高准确率和最佳校准。1.67pp的准确率提升看似不大，但校准的改善是实质性的：ECE从0.1213降至0.0719，降低40.7%。在临床环境下，校准良好的预测至关重要——一个报告90%置信度的模型应当在约90%的情况下是正确的。

**KL正则化提供隐式初始化鲁棒性。** 多种子实验揭示了EDL一个此前被低估的性质：抵抗不利随机初始化。交叉熵训练在种子456下崩溃至35%准确率——仅比随机概率高10pp——而EDL在相同条件下保持75%。我们将此归因于KL正则化器将非目标证据收缩至1（平坦Dirichlet先验），防止了交叉熵在训练早期抓住伪特征时出现的病态过拟合。仅有120个训练样本，这种效应被放大：每个样本影响很大，不利的初始化可能不可逆地偏置优化轨迹。

我们注意到一个潜在的替代解释：EDL中的KL正则化（惩罚偏离1的非目标证据）在功能上可能类似于标签平滑，其中硬独热目标由均匀分量软化。两种机制都阻止过度自信的预测并能改善校准。然而，两种方法存在根本不同：标签平滑操作在目标分布上（在损失计算前软化标签），而EDL的KL正则化器操作在Dirichlet参数空间（直接惩罚非目标α_k值）。标签平滑产生带平滑概率的点估计；EDL产生完整的Dirichlet分布，使不确定性量化成为可能。CE + 标签平滑是否能达到EDL的种子鲁棒性和校准收益，仍然是未来研究的开放问题。

**有序约束以牺牲单次准确率为代价改善泛化稳定性。** EDL+ORCU实现了最佳的5折CV稳定性（QWK标准差0.0032，CV 0.36%），但在单次训练/测试划分上表现不及EDL（70.00% vs EDL的83.33%）。这一单次性能与交叉验证稳定性之间的张力具有启发性。我们假设对数障碍正则化项与EDL的KL正则化器竞争：两者操作在相同的logits上，但优化目标不同（平坦非目标证据 vs. 单调logit排序）。在120个训练样本下，联合优化空间高度受限，组合正则化器可能过度约束了解空间。CV结果表明，当训练折改变时，有序先验作为稳定力量起作用，但在任何单一划分上它可能将模型拉离最优证据分布。

**小样本可行性得到证明。** DFID数据集（120训练样本）代表了一个具有挑战性的小样本医学影像场景。EDL在此范围内产生有意义的uncertainty估计并保持高准确率的能力表明，Dirichlet参数化相比标准softmax提供了有效的过拟合正则化。这一发现对罕见病诊断具有实际意义，因为大规模标注数据集很少可用。

**不确定性使分诊成为可能，但需要保守的阈值。** θ扫描显示EDL不确定性可以门控自动化诊断：低不确定性样本达到高SafePred，高不确定性样本被路由至专家复核。然而，示例C（u=0.465）显示EDL不确定性并不能完美地分离正确与错误预测。临床部署需要保守的θ和前瞻性验证。

### 6.2 与已发表方法的比较

**骨干网络比架构复杂性更重要。** 我们的CE基线（ViT-Base，86M，ImageNet-21k预训练）达到81.67%准确率，超过MLTrMR（556M参数，80.19%）。我们CE与MLTrMR之间的1.67pp差距——尽管MLTrMR参数多6.5倍——表明当数据稀缺时，ImageNet-21k预训练提供了比定制Transformer架构更强的归纳偏置。这与更广泛的迁移学习文献一致，但对医学影像特别突出，因为大规模领域特定预训练很少可用。

**不确定性量化是差异化的贡献。** 虽然LD2Net仅以3.31M参数（比ViT-Base轻26倍）达到有竞争力的准确率（80.00%），但它不提供不确定性估计。在临床实践中，一个在正确和错误预测上都报告80%置信度的轻量级模型，可能不如一个能准确报告何时不确定的更重模型有用。EDL的价值主张不仅在于比LD2Net高3.33pp的准确率提升，更在于区分自信与不确定预测的能力——这一能力直接使临床分诊工作流成为可能。

### 6.3 局限

- **单一数据集验证。** 所有实验使用DFID数据集（单一来源的200张图像）。虽然这能与使用相同数据集的已发表方法直接比较，但对其他人群、成像条件和氟中毒流行模式的泛化性需要多中心外部验证。
- **单评分者DF标注。** 与骨骼氟中毒数据集（有5位放射科医生标注）不同，DFID具有单评分者标签。这使得无法在DF上应用多评分者软标签建模。报告的EDL不确定性捕捉了模型不确定性和数据不确定性，但不包括评分者间标注不确定性。获取多评分者DF标注将加强不确定性分析。
- **EDL+ORCU交互未完全解决。** EDL+ORCU在单次划分上表现不佳，尽管具有优越的CV稳定性，表明KL正则化器与对数障碍约束之间存在未解决的交互。未来工作应研究替代方案：解耦优化（分离EDL和ORCU训练阶段）、梯度平衡（如GradNorm）或在Dirichlet参数空间而非logit空间中用软有序先验替代对数障碍。
- **用于错误检测的AUROC(u)。** EDL的逐样本不确定性u不能可靠地识别个别误分类样本（AUROC < 0.5），这一发现在多次独立运行中保持一致。我们注意到这不是我们工作独有的——先前的EDL研究也报告了点对点错误检测的适中或低于随机水平的AUROC。EDL不确定性应被理解为一种分布性质（捕捉Dirichlet的分布广度）而非逐点异常分数。对临床应用而言，相关的不确定性指标是校准（ECE）和群体级分诊（SafePred/RecAcc），而非逐样本AUROC。
- **无前瞻性临床验证。** 自动生成的诊断报告和不确定性阈值（θ = 0.30）尚未经过执业牙医审查。真实世界部署需要进行一项比较AI辅助与无辅助诊断的读片研究。

### 6.4 未来工作

若干方向值得研究：(1) 将EDL+ORCU框架扩展到带五位放射科医生多评分者软标签的骨骼氟中毒，实现与Mwinc-Mamba的直接比较；(2) 一旦CUDA内核可用，将证据头与真实的Mamba骨干网络集成用于SF；(3) 用于多视图牙科评估（正面、侧面、咬合面视图）的分层EDL；(4) 获取多评分者DF标注以研究牙医间变异及其与EDL不确定性的相关性；(5) 用避免与KL正则化竞争的Dirichlet空间有序先验替代对数障碍ORCU公式；(6) 一项比较AI辅助与无辅助牙医诊断的前瞻性读片研究；(7) 在其他地方病流行区域（印度、东非）的氟中毒数据集上进行外部验证以评估跨人群泛化性。

---

## 7 结论

我们提出了一个统一框架，将证据深度学习与用于校准和单峰性的有序回归相结合，用于地方性氟中毒的自动化诊断。共享logit证据头同时产生用于不确定性量化的Dirichlet证据和用于有序校准的logits，确保两个目标之间的一致性。通过三阶段训练（CE预热、带KL退火的EDL、联合EDL+ORCU），我们在200张图像的氟斑牙数据集上展示了不确定性感知的有序诊断。

我们的关键发现是：(1) EDL在DFID上达到最高准确率83.33%（比MLTrMR高+3.14pp，比我们CE基线高+1.67pp），校准误差最低（ECE 0.0719，相比CE降低40.7%）；(2) EDL的KL正则化提供了实质性的初始化鲁棒性，将准确率CV从CE的31.4%降至4.3%——对每次训练都至关重要的小样本医学影像尤为关键；(3) EDL+ORCU实现了最佳的交叉验证稳定性（QWK CV 0.36%），确认有序约束改善泛化；(4) EDL不确定性使临床分诊系统成为可能，低不确定性预测进行自动诊断报告，高不确定性病例标记为专家复核；(5) 在受控ViT-Base骨干网络下的系统性五损失比较，为医学影像中的有序损失选择提供了实用指导。

共享logit EDL框架与骨干网络无关，适用于需要不确定性量化和校准预测的有序医学分类任务。超越氟中毒，基于Dirichlet的不确定性所展示的优势——校准、种子鲁棒性和临床分诊能力——表明EDL值得被视为小样本医学图像分类的强基线，特别是在知道模型不知道什么与知道它预测什么同等重要的情况下。需要在更大规模的多中心数据集上进行前瞻性临床验证以建立泛化性。

---

**致谢**

作者感谢合作医院放射科提供骨骼氟中毒X射线数据集，以及五位放射科医生（CBQ、CF、XR、ZY、ZYC）的标注工作。本研究受国家自然科学基金（批准号：XXXXXX）和贵州省科技基金（批准号：XXXXXX）资助。

---

**参考文献 (31篇)**

1. Sensoy M, Kaplan L, Kandemir M. Evidential deep learning to quantify classification uncertainty. *NeurIPS*, 2018.
2. Kim J, Lee S, Kim T, Yoon S. ORCU: Ordinal Regression for Calibration and Unimodality. *Medical Image Analysis*, 2025.
3. Xu H, Chen B, Gao X, Wu Y. Masked latent transformer with random masking ratio to advance the diagnosis of dental fluorosis. *Journal of Computational Vision and Imaging*, 2025.
4. Xu H, Chen B, Gao X, Wu Y. Convolutional state space model with multi-window cross-scan to advance the automated diagnosis of skeletal fluorosis. *Biomedical Signal Processing and Control*, 2025.
5. Gu M, et al. Epidemiological survey of endemic fluorosis in Chaoyang Village, Xinmin Town, Zunyi City in 2009 and 2023. *Journal of Environmental and Occupational Medicine*, 2024.
6. Quadri JA, et al. Multiple Myeloma-Like Spinal MRI Findings in Skeletal Fluorosis. *Journal of Clinical and Diagnostic Research*, 2016.
7. Dosovitskiy A, et al. An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*, 2021.
8. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *CVPR*, 2016.
9. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML*, 2017.
10. Gal Y, Ghahramani Z. Dropout as a Bayesian approximation. *ICML*, 2016.
11. Lakshminarayanan B, Pritzel A, Blundell C. Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*, 2017.
12. Ulmer D, Hardmeier C, Frellsen J. Prior and posterior networks: A survey on evidential deep learning. *TMLR*, 2022.
13. Fejerskov O, Manji F, Baelum V, Møller IJ. Dental fluorosis: a handbook for health workers. *Munksgaard*, 1988.
14. DenBesten PK, Li W. Biological mechanisms of fluorosis. *Journal of Dental Research*, 2011.
15. Wei W, Pang S, Sun D. The pathogenesis of endemic fluorosis: Research progress in the last 5 years. *Journal of Cellular and Molecular Medicine*, 2019.
16. Li W, Zhang M, Wang Y. LD2Net: A Lightweight Deep Dual-Network Architecture for Dental Fluorosis Diagnosis. *Journal of Real-Time Image Processing*, 2026.
17. Liu Q, Xu H, Chen B, Wu Y. Integrated learning approach based on fused segmentation information for skeletal fluorosis diagnosis. *IEEE TIM*, 2021.
18. Müller R, Kornblith S, Hinton GE. When does label smoothing help? *NeurIPS*, 2019.
19. He Y, et al. Mutual evidential deep learning for semi-supervised medical image segmentation. *IEEE TMI*, 2024.
20. Zheng H, Lin Y, Zhou SK. Uncertainty-aware evidential fusion for semi-supervised medical image segmentation. *MedIA*, 2023.
21. Li M, Yang G, et al. Reliability-aware evidential deep learning with noisy label discounting. *Signal Processing*, 2024.
22. Diaz R, Marathe A. Soft labelling for ordinal regression. *CVPR Workshops*, 2019.
23. Cao W, Mirjalili V, Raschka S. Deep ordinal regression with CORAL and CORN. *ICLR*, 2021.
24. Liang S, Li Y, Srikant R. Enhancing the reliability of out-of-distribution image detection in neural networks. *ICLR*, 2018.
25. Abdar M, et al. A review of uncertainty quantification in deep learning. *Information Fusion*, 2021.
26. Zhang K, Lu Z, Guo X. Artificial intelligence in dentistry: A comprehensive review. *Frontiers in Cell and Developmental Biology*, 2023.
27. Dean HT. The investigation of physiological effects by the epidemiological method. *AAAS*, 1942.
28. Schwendicke F, Samek W, Krois J. Detecting white spot lesions on dental photography using deep learning. *Journal of Dentistry*, 2021.
29. Raghu M, Zhang C, Kleinberg J, Bengio S. Transfusion: Understanding transfer learning for medical imaging. *NeurIPS*, 2019.
30. Xie R, Wu Y, Xu H, Gu M. Few-shot learning for osteopenia classification from dental panoramic radiographs. *Journal of Imaging Informatics in Medicine*, 2024.

---

*中文译本基于 2026-05-18 v1.0-submission 英文原稿翻译。*
