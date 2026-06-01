---
date: 2026-05-14
author: Human
status: final
---
# 目标期刊投稿要求 — Medical Image Analysis

本文档供 WriteAgent 和 VizAgent 严格遵守，确保图表和文稿格式完全符合 Medical Image Analysis (MedIA, Elsevier, MICCAI Society Official Journal) 的投稿规范。

---

## 1. 基本信息

| 项目 | 要求 |
|------|------|
| 期刊全称 | Medical Image Analysis |
| 出版商 | Elsevier |
| 所属协会 | MICCAI Society |
| 投稿类型 | Full Original Research Paper |
| 排版系统 | LaTeX 优先（Word 也可，但需单栏排版） |
| LaTeX 模板 | `elsarticle` 文档类（[CTAN下载](https://mirrors.ctan.org/macros/latex/contrib/elsarticle.zip)） |
| 参考文献格式 | 作者-年份制（Author-Year），BibTeX 使用 `model2-names` 样式 + `\biboptions{authoryear}` |
| 首次提交格式 | 源文件（.tex 及图片文件）或 Word 文档，**不可提交 PDF 作为唯一源文件** |
| 修订稿提交 | 需提交带修改标记的版本（响应审稿意见时） |

---

## 2. 篇幅与字数限制

| 项目 | 限制 |
|------|------|
| Abstract | ≤ 250 词 |
| Keywords | 1–7 个 |
| Highlights | 3–5 条，每条 ≤ 85 个字符（含空格） |
| 正文字数 | 无严格上限，建议精炼，避免不必要冗长 |
| 图表数量 | 无硬性限制，建议正文图表总数 ≤ 10 张，补充材料不限 |

---

## 3. 论文结构要求（WriteAgent 严格遵守）

### 3.1 必需部分（按出现顺序）

1. **Title Page**（独立文件或置于手稿首页）
   - 论文标题（精炼、描述性强）
   - 全部作者全名（姓与名分开标注）
   - 通讯作者标注（* 号），并提供邮箱
   - 所有作者单位的完整邮寄地址（含国家）
   - 建议提供 ORCID（如有）

2. **Abstract**
   - ≤ 250 词
   - 陈述研究目的、主要方法、关键结果和重要结论
   - 避免引用参考文献和非标准缩写
   - 不使用 "In this paper..." 等冗余开场语

3. **Keywords**
   - 1–7 个关键词
   - 使用英文
   - 避免多词组合用 "and" 连接（应使用独立关键词）
   - 建议包含：fluorosis, deep learning, dental fluorosis, skeletal fluorosis, X-ray, classification 等

4. **Highlights**
   - 3–5 条核心研究发现
   - 每条 ≤ 85 个字符（含空格）
   - 单独文件提交
   - 使用项目符号列表，提供简洁结果陈述

5. **Introduction**
   - 清晰阐述临床背景与研究动机
   - 概述现有方法与不足
   - 明确列出本文贡献（通常 3–4 点）

6. **Methods**
   - 可细分为：Dataset、Preprocessing、Model Architecture、Training Strategy、Evaluation Metrics 等子节
   - 必须包含数据伦理声明（IRB 审批）
   - 必须包含可复现细节

7. **Results**
   - 可细分为：Experimental Setup、Comparison with State-of-the-Art、Ablation Study、Qualitative Analysis 等
   - 量化结果需注明指标名称和数值
   - 统计检验结果推荐提供（如 p 值、置信区间）

8. **Discussion**
   - 解释结果意义
   - 与现有方法比较
   - 讨论局限性与未来工作

9. **Conclusion**
   - 简短总结主要贡献与发现

10. **Declaration of Competing Interests**
    - 通过 Elsevier 官方 declarations tool 生成 .doc/.docx 文件单独上传
    - 所有作者须填写

11. **References**
    - 作者-年份制
    - 包含作者、标题、期刊名、卷号、页码、年份、DOI
    - 正文引用使用 (Author, Year) 格式

12. **Graphical Abstract**（必须提交）
    - 尺寸：高 531 pixels × 宽 1328 pixels（可读尺寸约 5 cm × 13 cm @ 96 dpi）
    - 格式：TIFF / EPS / PDF
    - 内容：简洁传达核心发现的图形概要
    - **注意**：MedIA 禁止使用生成式 AI 创建 Graphical Abstract，须用传统工具制作

### 3.2 可选但强烈建议的部分

- **Related Work**（可并入 Introduction 或独立成章）
- **Author Contributions (CRediT)**
  - 使用 CRediT 分类系统
  - 列出每位作者的具体贡献角色
- **Data Availability Statement**
  - 说明数据获取方式
  - 代码仓库链接（如有）
- **Funding Sources**
  - 按标准格式列出所有资助来源
  - 无资助须声明 "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."
- **Declaration of Generative AI Use**
  - 若在研究和写作中使用了生成式 AI，须在参考文献前新增一节声明
  - 说明所用工具名称、版本及具体用途
  - 示例内容："During the preparation of this work, the authors used Claude AI (Anthropic) to assist in literature organization and text polishing. All scientific content, data analysis, and conclusions were authored and verified by the human authors."
- **Supplementary Materials**
  - 与稿件同时提交
  - 正文中须明确引用
  - 每个补充文件需有描述性标题

---

## 4. 图表规范（VizAgent 严格遵守）

### 4.1 分辨率与像素要求

| 图像类型 | 最低 dpi | 单栏最小像素 (宽) | 双栏最小像素 (宽) |
|----------|----------|-------------------|-------------------|
| 彩色/灰度照片 (halftone) | 300 dpi | 1063 pixels | 2244 pixels |
| 线条图 (line art) | 1000 dpi | 3543 pixels | 7480 pixels |
| 组合图 (combined) | 500 dpi | 1772 pixels | 3740 pixels |

### 4.2 格式要求

| 项目 | 要求 |
|------|------|
| 位图格式 | TIFF / JPG（高质量）/ PNG |
| 矢量图格式 | EPS 或 PDF（字体必须嵌入或转为路径） |
| 严禁格式 | GIF、BMP、PICT、WPG、SWF 等 |
| 颜色模式 | RGB（用于电子版） |
| 图片文件命名 | `Figure_1.tif`, `Figure_2.pdf` 等，清晰对应正文引用 |

### 4.3 尺寸与布局

- **单栏宽度**：约 80–90 mm
- **双栏（整页）宽度**：约 170–180 mm
- 图表应自适应上述宽度，避免缩放导致信息丢失
- 同一论文中同类图表保持统一尺寸

### 4.4 字体与标注

- 图表内文字尽量最少
- 必要文字使用 8–10 pt 字号
- 所有符号、缩写须在图注/表注中解释
- 字体推荐 Arial / Helvetica
- 避免因图片尺寸过大导致文字相对图像比例过小

### 4.5 配色

- 推荐使用色盲友好调色板（如 ColorBrewer Set2）
- 避免仅靠红/绿区分信息
- 确保灰度打印时仍可区分

### 4.6 图注要求

- 每个图表必须有独立图注
- 图注集中放在 LaTeX 源文件的 `\caption{}` 中
- 图注首句为简短标题，后续为必要说明
- 所有缩写和符号在图注中展开

### 4.7 表格要求

- 表格须为可编辑文本，**不可提交为图片**
- 连续编号，每个表格有表题
- 避免使用垂直分割线
- 避免单元格底纹（阴影）
- 表格内容不与正文重复
- 使用三线表（顶线、表头下线、底线）

### 4.8 严禁事项

- **禁止使用生成式 AI 创建或修改任何投稿图像**（包括但不限于图形摘要、封面图、正文图表）
- 唯一例外：若 AI 图像生成属于研究方法本身（如用 GAN 生成训练数据），此种情况可接受
- 所有图表须使用传统科学绘图工具生成（Matplotlib、Python、R、MATLAB、Origin 等）

---

## 5. 引用规范

| 项目 | 要求 |
|------|------|
| 引用格式 | 作者-年份制（Author-Year / Harvard） |
| LaTeX 配置 | `\bibliographystyle{model2-names}` + `\biboptions{authoryear}` |
| 条目完整性 | 作者、标题、期刊名全称、卷号、页码、出版年份、DOI |
| 正文引用 | (Author, Year) 或 Author (Year) |
| 参考文献管理 | Paperpile、EndNote、Zotero、Mendeley 均提供 MedIA 专属样式 |
| DOI 要求 | 强烈建议为每条参考文献提供 DOI |

---

## 6. 伦理与合规声明

| 项目 | 要求 |
|------|------|
| 人体受试者研究 | 须在 Methods 中声明 IRB/Ethics Committee 审批信息 |
| 知情同意 | 涉及患者数据须声明已获取知情同意 |
| 数据隐私 | 所有患者信息须脱敏处理 |
| 利益冲突 | 通过 Elsevier declarations tool 统一提交 |
| AI 使用声明 | 若使用 AI 工具（包括 Claude），须单独声明 |
| 预印本 | 允许在投稿前发布预印本（如 arXiv） |
| 代码与数据 | 鼓励提供代码和数据获取链接 |

---

## 7. 首次提交流程要点

1. 准备手稿源文件（.tex 或 .doc/.docx）
2. 准备全部高分辨率图片文件（单独文件，不嵌入 Word）
3. 准备 Graphical Abstract（必须）
4. 准备 Highlights（必须）
5. 准备 Declaration of Competing Interests 文件（通过 Elsevier 工具生成）
6. 准备 Cover Letter（单独文件）
7. 建议在提交系统中填写审稿人建议/排除列表
8. 所有文件通过 Elsevier Editorial System (EES) 提交

---

*本文件内容基于 Medical Image Analysis 官方 Guide for Authors（Elsevier, 2026），所有 Agent 必须以本文档为格式最高准则。*