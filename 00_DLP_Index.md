---
tags: [DLP, index, MOC]
date: 2026-04-08
---

# DLP 主題索引 (Map of Content)

> 本索引收錄 DLP（Deep Learning Practice）資料夾內所有筆記，依類型分組。

---

## 論文筆記（Paper Notes）

| 筆記 | 核心概念 | 來源 |
|------|---------|------|
| [[MaskGIT]] | Masked Generative Image Transformer：雙向 Transformer + 迭代解碼，比 autoregressive 快 30–64× | arXiv:2202.04200 |

---

## 實驗報告（Lab Reports）

| 筆記 | 說明 |
|------|------|
| [[DL_LAB3_314553044_楊正豪 1]] | Lab 3 正式報告：實作 MaskGIT 進行 Image Inpainting，含 Multi-head Attention、Transformer Stage 2、Iterative Inpainting |

---

## 實作程式碼筆記（Implementation Notes）

| 筆記 | 說明 |
|------|------|
| [[Lab3]] | MaskGIT 實作細節筆記：encode_to_z、mask token、transformer logits 流程，含實驗截圖 |

---

## 概念地圖

```
VQGAN（Stage 1 Tokenizer）
  ↓  壓縮圖片 → 離散 visual tokens
MaskGIT Transformer（Stage 2）
  ↓  雙向 self-attention，MVTM 訓練
Iterative Decoding
  ↓  8 輪迭代，cosine mask schedule，confidence-based 保留
Image Inpainting / Generation
```

> **核心論文 → 實作對照**：
> - [[MaskGIT]]（理論）← → [[DL_LAB3_314553044_楊正豪 1]]（報告）← → [[Lab3]]（程式碼筆記）

---

## 孤立筆記（Orphan Notes）

> 以下筆記目前未被其他筆記引用，特別標出以避免遺漏：

- [[Lab3]]（實作程式碼筆記，內容較簡略，可搭配 [[MaskGIT]] 對照閱讀）

---

## 相關資源

| 類型 | 檔案 |
|------|------|
| 論文 PDF | `Maskgit.pdf`（在 DLP 根目錄） |
| 課程綱要 | `Syllabus 2026 spring for NYCU.pdf` |

---

*最後更新：2026-04-08*
