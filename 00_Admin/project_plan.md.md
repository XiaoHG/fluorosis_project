---
date: 2026-05-14
author: Human
status: final
---
# 项目计划书：深度学习 + 氟中毒多 Agent 论文协作

## 项目概览

| 项目   | 内容                                       |
| ---- | ---------------------------------------- |
| 课题方向 | 深度学习在氟中毒、氟骨症、氟斑牙诊断中的应用                   |
| 预期产出 | 一篇可投稿至 Medical Image Analysis 的原创研究论文    |
| 协作模式 | 7 个专职 Claude Agent + 人类项目经理              |
| 总周期  | 12 周（2026 年 5 月 13 日 – 2026 年 8 月 4 日）   |
| 目标期刊 | Medical Image Analysis (MedIA, Elsevier) |

---

## 一、阶段划分与时间表

### 阶段 0：环境与知识准备（第 1 周：5.13 – 5.19）

**目标**：建立领域知识底座，初始化项目环境，完成文献初筛。

| 任务 ID | 任务描述                        | 负责人         | 交付物                                                  |
| ----- | --------------------------- | ----------- | ---------------------------------------------------- |
| P0-1  | 撰写领域基础文档                    | 👤 人类       | `01_Knowledge_Base/` 下 4 个文件                         |
| P0-2  | 撰写目标期刊要求文档                  | 👤 人类       | `00_Admin/journal_requirements.md`                   |
| P0-3  | 撰写项目计划书                     | 👤 人类       | `00_Admin/project_plan.md`                           |
| P0-4  | 初始化项目目录结构与 Git 仓库           | 👤 人类       | 完整目录骨架                                               |
| P0-5  | 部署 LitAgent System Prompt   | 👤 人类       | `00_Admin/agent_prompts/lit_agent_system_prompt.md`  |
| P0-6  | LitAgent 执行文献初筛（检索 20-30 篇） | 🤖 LitAgent | `02_Literature/01_initial_search/literature_list.md` |

**里程碑 M0**（5.19）：知识底座就绪，文献初筛列表产出。

---

### 阶段 1：文献深挖（第 2 周：5.20 – 5.26）

**目标**：完成精读与缺口分析，为创新点提供扎实文献基础。

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P1-1 | 精读 Top 10 文献，生成深度摘要 | 🤖 LitAgent | `02_Literature/02_deep_read/` 下 10 个 .md |
| P1-2 | 制作核心文献对比表 | 🤖 LitAgent | `02_Literature/03_comparison_table.md` |
| P1-3 | 撰写研究缺口分析 | 🤖 LitAgent | `02_Literature/04_gap_analysis.md` |
| P1-4 | 生成 README.md | 🤖 LitAgent | `02_Literature/README.md` |
| P1-5 | 人类审核 LitAgent 全部产出 | 👤 人类 | 反馈或批准 |

**里程碑 M1**（5.26）：文献分析阶段完成，缺口明确。

---

### 阶段 2：创新点凝练（第 3–4 周：5.27 – 6.9）

**目标**：基于文献缺口，生成并选定创新点。

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P2-1 | 部署 InnoAgent System Prompt | 👤 人类 | — |
| P2-2 | 生成研究趋势雷达图（文字版） | 🤖 InnoAgent | `03_Innovation/01_radar_chart.txt` |
| P2-3 | 生成 3 个创新点提案 | 🤖 InnoAgent | `03_Innovation/02_idea_proposals/` 下 3 个 .md |
| P2-4 | 🔴 人类评估 3 个提案，选定主攻创新点 | 👤 人类 | `11_Decision_Logs/2026-06-0X_选定创新点.md` |
| P2-5 | 精炼选定创新点，绘制技术路线图 | 🤖 InnoAgent | `03_selected_idea.md`, `04_tech_roadmap.md` |

**里程碑 M2**（6.9）：创新点锁定，技术路线明确。

---

### 阶段 3：模型框架设计（第 5–6 周：6.10 – 6.23）

**目标**：将创新点转化为具体网络结构和训练策略。

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P3-1 | 部署 ModAgent System Prompt | 👤 人类 | — |
| P3-2 | 设计 2 个备选模型架构 | 🤖 ModAgent | `04_Model_Design/01_architecture_variants/` 下 2 个 .md |
| P3-3 | 🔴 人类选择最终架构 | 👤 人类 | `11_Decision_Logs/2026-06-XX_确认模型架构.md` |
| P3-4 | 完善选定架构，输出损失函数和训练配置 | 🤖 ModAgent | `02_selected_architecture.md`, `03_loss_functions.md`, `04_training_config.md` |

**里程碑 M3**（6.23）：模型架构冻结。

---

### 阶段 4：实验设计（第 7 周：6.24 – 6.30）

**目标**：制定完整、可复现的实验方案。*（必须在 ModAgent 完成全部输出后启动）*

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P4-1 | 部署 ExpDesignAgent System Prompt | 👤 人类 | — |
| P4-2 | 设计数据集划分方案 | 🤖 ExpDesignAgent | `05_Exp_Design/01_dataset_split.md` |
| P4-3 | 设计数据增强策略 | 🤖 ExpDesignAgent | `05_Exp_Design/02_augmentation.md` |
| P4-4 | 定义评价指标 | 🤖 ExpDesignAgent | `05_Exp_Design/03_metrics.md` |
| P4-5 | 确定对比方法与复现方案 | 🤖 ExpDesignAgent | `05_Exp_Design/04_baselines.md` |
| P4-6 | 设计消融实验矩阵 | 🤖 ExpDesignAgent | `05_Exp_Design/05_ablation_plan.md` |
| P4-7 | 规划可视化需求清单 | 🤖 ExpDesignAgent | `05_Exp_Design/06_vis_requirements.md` |
| P4-8 | 🔴 人类审核实验方案 | 👤 人类 | `11_Decision_Logs/2026-06-XX_批准实验方案.md` |

**里程碑 M4**（6.30）：实验方案锁定，可进入实施。

---

### 阶段 5：实验实施（第 8–10 周：7.1 – 7.21）

**目标**：完成所有实验，获得全部定量结果。

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P5-1 | 部署 ImplAgent System Prompt | 👤 人类 | — |
| P5-2 | 生成完整代码框架（模型、数据加载、训练脚本） | 🤖 ImplAgent | `06_Implementation/code/` |
| P5-3 | 生成全部实验配置文件 | 🤖 ImplAgent | `06_Implementation/configs/` |
| P5-4 | 生成运行说明 | 🤖 ImplAgent | `06_Implementation/README.md` |
| P5-5 | 👤 人类执行训练（完整模型 + 对比方法 + 消融实验） | 👤 人类 | 训练日志、模型权重 |
| P5-6 | 👤 人类执行评估，汇总结果 | 👤 人类 | `06_Implementation/results/` 下各 CSV |
| P5-7 | 结果汇总与完整性检查 | 🤖 ImplAgent + 👤 | 确认所有必需结果已产出 |

**里程碑 M5**（7.21）：全部实验数据产出。

---

### 阶段 6：图表制作（第 11 周前半：7.22 – 7.25）

**目标**：生成符合 MedIA 规范的高清图表及图注。

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P6-1 | 部署 VizAgent System Prompt | 👤 人类 | — |
| P6-2 | 为每个图表生成绘图脚本和效果说明 | 🤖 VizAgent | `07_Visualization/01_drafts/` 下脚本 + README.md |
| P6-3 | 👤 人类运行脚本，生成初版图表 | 👤 人类 | `01_drafts/` 下 .png 预览图 |
| P6-4 | 🔴 人类审阅草图，提出修改意见 | 👤 人类 | 反馈 |
| P6-5 | 根据反馈修改脚本，生成终版图表脚本 | 🤖 VizAgent | `04_plotting_scripts/` |
| P6-6 | 👤 人类运行终版脚本，生成高清图表 | 👤 人类 | `02_final_figures/` 下 .pdf + .png |
| P6-7 | 生成图注草稿 | 🤖 VizAgent | `03_captions/figure_captions.md` |

**里程碑 M6**（7.25）：图表全部就绪。

---

### 阶段 7：论文撰写与审核（第 11 周后半 – 第 12 周：7.26 – 8.4）

**目标**：完成论文撰写、多轮审核及投稿材料准备。

| 任务 ID | 任务描述 | 负责人 | 交付物 |
|---------|---------|--------|--------|
| P7-1 | 部署 WriteAgent System Prompt | 👤 人类 | — |
| P7-2 | 按 MedIA 结构撰写论文初稿 | 🤖 WriteAgent | `08_Manuscript/v1_first_draft/manuscript.tex` + PDF |
| P7-3 | 生成 Highlights（3-5 条，≤85 字符/条） | 🤖 WriteAgent | `08_Manuscript/highlights.md` |
| P7-4 | 生成 Graphical Abstract 设计方案（文字描述） | 🤖 WriteAgent | `08_Manuscript/graphical_abstract_plan.md` |
| P7-5 | 生成 Declaration of Generative AI Use 草稿 | 🤖 WriteAgent | `08_Manuscript/ai_declaration.md` |
| P7-6 | 生成 CRediT 作者贡献声明草稿 | 🤖 WriteAgent | `08_Manuscript/credit_contributions.md` |
| P7-7 | 第一轮审核：逻辑与结构 | 🤖 WriteAgent | `09_Review/01_logic_review.md`，修订稿 v2 |
| P7-8 | 第二轮审核：技术细节 | 🤖 WriteAgent | `09_Review/02_technical_review.md`，修订稿 v3 |
| P7-9 | 第三轮审核：语言与格式（含 MedIA 检查清单） | 🤖 WriteAgent | `09_Review/03_language_review.md`，终稿 |
| P7-10 | 👤 人类 + 合作者终审（重点：医学表述） | 👤 人类 | `09_Review/04_external_comments.md` |
| P7-11 | 根据外部意见最终修改 | 🤖 WriteAgent | `08_Manuscript/manuscript.tex`（终稿） |
| P7-12 | 👤 人类制作 Graphical Abstract（非 AI） | 👤 人类 | `10_Submission/graphical_abstract.pdf` |
| P7-13 | 👤 整理投稿包 | 👤 人类 | `10_Submission/` 下全部文件 |

**里程碑 M7**（8.4）：论文定稿，投稿包就绪，可提交。

---

## 二、关键决策节点

| 编号 | 时间 | 决策内容 | 决策者 |
|------|------|----------|--------|
| D1 | 第 3 周末（6.2 前） | 从 3 个创新点提案中选择主攻方向 | 👤 人类 |
| D2 | 第 6 周中（6.16 前） | 从 2 个备选架构中确认最终模型 | 👤 人类 |
| D3 | 第 7 周末（6.30） | 审核并批准实验方案 | 👤 人类 |
| D4 | 第 11 周初（7.23） | 审阅图表草图，提出修改意见 | 👤 人类 |
| D5 | 第 12 周末（8.4） | 终审全文，签署定稿 | 👤 人类 |

所有决策记录存入 `11_Decision_Logs/`，命名格式：`YYYY-MM-DD_决策主题.md`。

---

## 三、Agent 部署时序

| Agent | System Prompt 文件 | 首次启动时间 | 启动于阶段 |
|-------|-------------------|-------------|-----------|
| LitAgent | `lit_agent_system_prompt.md` | 第 1 周 | 阶段 0 |
| InnoAgent | `inno_agent_system_prompt.md` | 第 3 周 | 阶段 2 |
| ModAgent | `mod_agent_system_prompt.md` | 第 5 周 | 阶段 3 |
| ExpDesignAgent | `exp_agent_system_prompt.md` | 第 7 周 | 阶段 4 |
| ImplAgent | `impl_agent_system_prompt.md` | 第 8 周 | 阶段 5 |
| VizAgent | `viz_agent_system_prompt.md` | 第 11 周 | 阶段 6 |
| WriteAgent | `write_agent_system_prompt.md` | 第 11 周 | 阶段 7 |

---

## 四、风险预案

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 数据量不足/标注质量差 | 模型训练困难 | ExpDesignAgent 预设过采样/GAN 生成/迁移学习方案；必要时与临床合作者协商补充标注 |
| 模型不收敛或性能不佳 | 实验周期延长 | ImplAgent 提供简单基线快速验证；人类可调整超参数；InnoAgent 评估降级方案 |
| 医学审查认为缺乏可解释性 | 论文被质疑 | ModAgent 架构中已预设可解释性出口；VizAgent 重点输出 Grad-CAM 可视化 |
| 写作质量不达标 | 投稿被拒 | WriteAgent 分三轮审核，人类合作者提前介入医学部分 |
| 某 Agent 产出不符合预期 | 下游阻塞 | 人类及时介入，提供更具体指令或手动修正中间产物；关键阶段设置人类审核点 |
| 生成式 AI 使用不合规 | MedIA Desk Reject | VizAgent 严格仅输出绘图脚本；WriteAgent 生成 Declaration 声明；所有图表由人类用传统工具生成 |
| 数据隐私/伦理问题 | 投稿被拒 | Methods 中声明 IRB 审批和知情同意；数据脱敏；Declaration 通过 Elsevier 工具生成 |

---

## 五、沟通与协同规则

1. **传递标准**：所有 Agent 输出必须遵循 YAML 元数据头部，注明 `input_from` 和 `output_to`。
2. **版本控制**：每完成一个阶段，进行一次 Git commit 并打 tag（如 `v1.0_lit_review_done`）。
3. **人类职责**：
   - 在 D1–D5 关键节点做出决策并记录
   - 负责医学事实核查和伦理合规
   - 执行代码训练和图表生成脚本
   - 处理非技术性内容（致谢、利益冲突声明、Cover Letter 等）
   - 制作 Graphical Abstract（非 AI 工具）
4. **外部反馈**：导师或临床合作者的批注统一存入 `09_Review/04_external_comments.md`，由 WriteAgent 统一处理。
5. **文件交换**：所有 Agent 间传递通过共享目录结构完成，不依赖对话上下文。