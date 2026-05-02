---
tags: [notes, DLP, CV, generative-model, transformer]
date: 2026-04-02
source: Maskgit.pdf
paper: "MaskGIT: Masked Generative Image Transformer (arXiv:2202.04200)"
authors: "Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, William T. Freeman (Google Research)"
wiki: [[wiki/Transformer]], [[wiki/Diffusion_Model]]
---

# MaskGIT: Masked Generative Image Transformer

## 這章在幹嘛

傳統的 autoregressive image generation（如 VQGAN）把圖片壓扁成一個 1D 序列，然後從左到右、從上到下一個一個 token 生成，就像在「打字」。這樣做有兩個致命問題：

1. **慢**：生成 256×256 的圖需要跑 256 次 forward pass，512×512 更高達 1024 次
2. **單向**：每個 token 只能看到它「左邊和上面」的 token，看不到右邊和下面——但圖片本來就沒有方向性

MaskGIT 的答案是：**所有 token 一起生成，然後迭代精修**。

用 BERT 在 NLP 的成功做法搬到影像生成：訓練時預測被 [MASK] 蓋掉的 token，推理時從全遮蓋開始，每輪平行預測所有未知 token，只保留最有把握的，再反覆細化，**8 輪就搞定**，比 autoregressive 快 30–64 倍。

---

## 比喻

想像一個畫家和一個打字員的差別：

- **VQGAN（打字員）**：從畫布左上角開始，一個格子一個格子填，填完才能看整體——右邊還沒畫，所以不能根據右邊調整左邊。
- **MaskGIT（畫家）**：先輕輕勾勒整體輪廓（第一輪），然後每輪加更多細節，最後得到完整高清圖。每一輪的決策都能看到整張畫布的現狀。

另一個比喻：這就像**填字遊戲**。一開始全部空白，你先填最有把握的那幾格，然後靠已填的格子猜其他格，越填越容易，而不是硬規定從第一格開始填。

---

## 核心概念

### 兩階段架構

MaskGIT 沿用業界標準的 two-stage 設計：

**Stage 1：Tokenizer（影像量化）**

> 把連續像素空間壓縮成一張離散 token 地圖，就像把高清照片存成文字代碼。

採用 VQGAN 的架構，三個核心組件：
- **Encoder** $E$：把影像 $x \in \mathbb{R}^{H \times W \times 3}$ 編碼成連續 embedding $E(x)$
- **Codebook** $\{e_k\}_{k=1}^{K}$：$K=1024$ 個離散碼字，透過 nearest-neighbor lookup 把 embedding 量化成 visual tokens
- **Decoder** $G$：把 token 序列還原成像素 $\hat{x}$

壓縮比固定為 16×，即 $256 \times 256$ 圖像 → $16 \times 16 = 256$ 個 tokens；$512 \times 512$ → $32 \times 32 = 1024$ tokens。

**Stage 2：Bidirectional Transformer（MaskGIT 的創新所在）**

> 使用雙向 self-attention，讓每個 token 都能「看到」圖上其他所有位置，而不是只看到左上方。

---

### Masked Visual Token Modeling（MVTM）

訓練目標和 BERT 幾乎一樣，差別在於這裡預測的是 visual token 而非文字。

設 $Y = [y_i]_{i=1}^{N}$ 為 token 序列，$M = [m_i]_{i=1}^{N}$ 為對應的 binary mask。
- $m_i = 1$：位置 $i$ 被 [MASK] 遮蓋
- $m_i = 0$：位置 $i$ 保持原 token 不動

訓練時從 mask scheduling function $\gamma(\cdot)$ 採樣一個比例 $r \in [0, 1)$，然後隨機選 $\lfloor \gamma(r) \cdot N \rfloor$ 個位置放 [MASK]。

損失函數：

$$\mathcal{L}_{\text{mask}} = -\mathbb{E}_{Y \sim \mathcal{D}} \sum_{\substack{i \in [1,N] \\ m_i = 1}} \log p(y_i | Y_M)$$

逐項解釋：
- $\sum_{i: m_i=1}$：只對被遮蓋的位置計算 loss，沒遮蓋的不管
- $\log p(y_i | Y_M)$：給定所有可見 token（包含雙向 context）後，正確預測第 $i$ 個 token 的 log 機率
- 與 autoregressive 的關鍵差異：$Y_M$ 包含**所有方向**的上下文，不只是左側

---

## 推導

### Iterative Decoding（推理階段）

> 從空白畫布出發，每輪預測 + 保留 + 再遮蓋，直到整張圖填滿。

設總迭代次數為 $T$（實驗中 $T=8$），以下是第 $t$ 輪的流程：

**步驟 1 — Predict（平行預測）**

給定當前 masked token 序列 $Y_M^{(t)}$，模型一次跑 forward pass，輸出所有被遮蓋位置的 probability distribution：

$$p^{(t)} \in \mathbb{R}^{N \times K}$$

- $N$：token 總數，$K$：codebook 大小（1024）
- 這一步是 MaskGIT 的核心優勢：**一次 forward pass 同時預測所有位置**，而 autoregressive 需要一次只預測一個

**步驟 2 — Sample（採樣）**

對每個被遮蓋位置 $i$，根據 $p_i^{(t)} \in \mathbb{R}^K$ 採樣 token $\hat{y}_i^{(t)}$，並記錄其 confidence score（即採樣到的 token 的機率值）。

未被遮蓋的位置 confidence score 設為 $1.0$（確定已知）。

**步驟 3 — Mask Schedule（計算下一輪的 mask 數量）**

$$n = \left\lfloor \gamma\!\left(\frac{t}{T}\right) \cdot N \right\rfloor$$

直覺：$t/T$ 代表解碼進度，$\gamma(\cdot)$ 決定還需要遮蓋幾個 token——越到後期越少。

**步驟 4 — Mask（保留最有把握的，其餘再遮蓋）**

從所有預測的 token 中，保留 confidence 最高的那些，把其餘 $n$ 個再次遮蓋：

$$m_i^{(t+1)} = \begin{cases} 1 & \text{if } c_i < \text{sorted}_j(c_j)[n] \\ 0 & \text{otherwise} \end{cases}$$

直覺：**你最有把握的先定稿**，不確定的讓下一輪再看更多上下文後重新決定。這像剪輯影片——先敲定主要鏡頭，再補細節。

---

### Mask Scheduling Function

這是 MaskGIT 最關鍵的設計之一。函數 $\gamma: [0,1] \to (0,1]$ 必須滿足：
1. 連續、有界
2. 單調遞減，$\gamma(0) = 1$，$\gamma(1) = 0$（確保收斂）

論文比較三大類型：

| 類型 | 例子 | 直覺 |
|------|------|------|
| Linear | $\gamma(r) = 1 - r$ | 每輪等量解遮 |
| **Concave**（凹） | **Cosine**, Square, Cubic, Exponential | 開始慢慢解、最後快速填完 |
| Convex（凸） | Square root, Logarithmic | 開始快速定案、後面細調 |

**Cosine schedule（最佳）：**
$$\gamma(r) = \cos\!\left(\frac{\pi}{2} \cdot r\right)$$

為什麼 cosine 最好？直覺上，圖像生成是個「less-to-more information」的過程：
- 最初 token 幾乎全空，模型只需要自信地猜幾個「最顯眼」的地方（輪廓）
- 到後期，大部分 token 已定案，context 豐富，模型可以快速精確地填完細節

Cosine 函數的形狀恰好符合這個節奏：開頭平坦（慢慢解遮），中後段急降（快速填完）。

---

## 小例子

### 生成一張 16×16 = 256 token 的圖

設 $T=8$，cosine schedule。

| 輪次 $t$ | 進度 $t/T$ | $\gamma(t/T)$ | 剩餘 mask 數 |
|---------|-----------|----------------|------------|
| 0（初始） | 0 | 1.00 | 256 |
| 1 | 0.125 | 0.98 | 251 |
| 2 | 0.25 | 0.92 | 236 |
| 3 | 0.375 | 0.83 | 212 |
| 4 | 0.5 | 0.71 | 181 |
| 5 | 0.625 | 0.56 | 144 |
| 6 | 0.75 | 0.38 | 98 |
| 7 | 0.875 | 0.19 | 49 |
| 8（完成） | 1.0 | 0.00 | 0 |

可以看到解遮速度是先慢後快，最後幾輪一口氣填完大半。

---

## 易混淆

### MaskGIT vs BERT（masked language modeling）

| | BERT | MaskGIT |
|--|------|---------|
| 任務 | 語言 representation learning | 影像 **generation** |
| Mask ratio | 固定 15% | 從 100% 到 0% 動態變化 |
| Inference | 一次預測就完成（填空） | 多輪迭代精修（從無到有） |
| 目標 | 學好 embedding 供下游使用 | 直接生成高品質圖像 |

BERT 不需要從空白生成——它做的是「填空」。MaskGIT 才需要「從零創造」，所以需要一套完整的 mask scheduling 策略讓模型學會在各種 masking 比例下都能運作。

### Autoregressive（VQGAN）vs MaskGIT

| | VQGAN（autoregressive） | MaskGIT |
|--|------------------------|---------|
| 生成順序 | 固定：從左到右、從上到下 | 自由：最有把握的優先 |
| Attention 方向 | 單向（causal） | 雙向 |
| 生成步數 | = token 數（256 或 1024） | 固定 8 步 |
| 速度 | 慢（sequential） | 快 30–64× |
| 圖像編輯 | 困難（打破 raster order） | 簡單（直接修改 mask） |

---

## 重點總結

1. **MVTM 訓練**：用 bidirectional transformer，預測隨機 masked visual tokens，mask ratio 從訓練時的 $(0, 1)$ 均勻採樣
2. **Iterative decoding**：$T=8$ 步，每步平行預測所有 token，只保留 confidence 最高的，其餘重新 mask
3. **Cosine mask schedule** 是關鍵設計，比 linear/convex/其他 concave 都更好
4. **速度**：比 VQGAN autoregressive decoding 快 30–64×（token 數越多加速越大）
5. **品質**：ImageNet 256×256 FID 6.18（vs VQGAN 15.78），512×512 FID 7.32（新 SOTA）
6. **彈性**：Image inpainting、outpainting、class-conditional editing 不需重新訓練，直接調整初始 mask 即可

---

## 研究前沿與實務

### 與其他方向的對比

| 方向 | 代表 | 優勢 | 劣勢 |
|------|------|------|------|
| GAN | BigGAN | 速度快、品質高 | mode collapse、訓練不穩定 |
| Diffusion | ADM | 多樣性高 | 250 步，慢 |
| Autoregressive | VQGAN | 訓練穩定、MLE | 生成慢、單向 |
| **Masked** | **MaskGIT** | **快 + 多樣 + 可編輯** | 需要精心設計 mask schedule |

### MaskGIT 的侷限

- **Stage 1 的上限**：生成品質仍受 VQGAN tokenizer 的重建品質限制，有失真
- **Codebook size**：目前 1024 碼字，可能不足以表達極高解析度的細節
- **Temperature tuning**：採樣時需要 temperature annealing 才能兼顧多樣性與品質

### 後續延伸

- **MaskBit**（2024）：改進 tokenizer，採用 binary codebook
- **MAR**（Masked Autoregressive）：結合 diffusion-based decoder，在 continuous token space 做 masked generation
- 目前 masked generation 已成為 image generation 主流範式之一，與 diffusion 並立
