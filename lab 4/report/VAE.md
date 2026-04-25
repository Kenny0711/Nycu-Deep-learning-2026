---
tags: [paper, CV, video-generation, VAE, LSTM]
date: 2026-04-14
source: arXiv:1802.07687
---

[VAE影片](https://www.bilibili.com/video/BV1LgDzYME7c/?spm_id_from=333.337.search-card.all.click&vd_source=8332afe5679bc5ca106c4926d4ea32d0)
![[Pasted image 20260413204813.png]]
![[Pasted image 20260414215531.png]]

---

# Stochastic Video Generation with a Learned Prior (SVG)

> Emily Denton & Rob Fergus — NYU / Facebook AI Research — ICML 2018

## 一句話

把 VAE 的 ELBO 框架推廣到影片序列，用**時間相依的隨機 latent variable** 捕捉未來的不確定性；核心亮點是 Learned Prior——一個學會「什麼時候世界是隨機的」的先驗網路。

---

## 動機

**為什麼現有方法不行？**

預測未來影片幀時，世界本質上是隨機的（球打到牆、機械臂偏移、關節位置抖動）。

| 方法 | 問題 |
|------|------|
| 確定性模型（L2 loss） | 對多種可能的未來取平均 → **畫面模糊（blurry）** |
| GAN | 訓練不穩定、mode collapse → 分布覆蓋不完整 |
| 固定先驗 VAE（Babaeizadeh 2017） | 所有時間步用同一分布採樣，忽略時間相依性；訓練需要三階段 |

這篇論文的切入點：**不確定性本身是時間相依的**。球飛行中軌跡確定；打到牆的瞬間才是隨機的。應該讓模型學會這件事，而不是一直注入固定量的雜訊。

---

## 方法

### 模型骨架

兩個核心組件在每個時間步 $t$ 運作：

$$\text{Prediction model:}\quad \hat{\mathbf{x}}_t = \mu_\theta(\mathbf{x}_{1:t-1},\, \mathbf{z}_{1:t})$$

$$\text{Inference network (訓練時):}\quad q_\phi(\mathbf{z}_t \mid \mathbf{x}_{1:t}) = \mathcal{N}(\mu_\phi(\mathbf{x}_{1:t}),\, \sigma_\phi(\mathbf{x}_{1:t}))$$

$\mathbf{z}_t$ 攜帶的是「過去幀無法確定、但當前幀才有的隨機資訊」。

> [!info]+ Figure 1：Inference 與 Generation 的圖形模型
> ![[image/papers/svg-2018/fig-p3-1.png]]
>
> 左：訓練時 inference network 從 $x_{1:t}$ 估計後驗；中：SVG-FP 用固定先驗生成；右：SVG-LP 用 learned prior 生成。

### SVG-FP：Fixed Prior

$$p(\mathbf{z}_t) = \mathcal{N}(\mathbf{0}, \mathbf{I})$$

每個時間步都從標準高斯採樣。簡單但盲目——不知道現在是否是需要隨機的時刻，等於在每一步都加固定量的雜訊。

### SVG-LP：Learned Prior ⭐

先驗本身也是神經網路（$\text{LSTM}_\psi$），根據過去幀預測當前時刻的不確定性：

$$p_\psi(\mathbf{z}_t \mid \mathbf{x}_{1:t-1}) = \mathcal{N}(\mu_\psi(\mathbf{x}_{1:t-1}),\, \sigma_\psi(\mathbf{x}_{1:t-1}))$$

- $\sigma_\psi$ 小 → 模型相信未來是確定的，latent 不重要
- $\sigma_\psi$ 大 → 模型預測到高度不確定事件，latent 攜帶大量隨機資訊

訓練時 KL divergence 改為拉近 $q_\phi$ 和 $p_\psi$（而非 $\mathcal{N}(0,I)$），讓 prior 學會追蹤 posterior 的分布。

> [!abstract]+ 架構圖：SVG-LP 一個時間步的資料流
> ![[image/excalidraw/svg-2018/svg-lp-architecture.excalidraw.md]]
>
> 三條路徑：Inference network（上）估計後驗 → Learned Prior（中）預測先驗 → 兩者 KL；採樣 $z_t$ 後 Prediction model（下）生成 $\hat{x}_t$ → L2 Loss。

### Skip Connections

從最後一個 ground truth 幀的 encoder 輸出直接 skip 到 decoder，讓靜態背景可以直接複製。模型只需要專注在「與前一幀不同的部分」。

### 前向計算（訓練，一個時間步）

```
# Inference
h_t      = Enc(x_t)
μ_φ, σ_φ = LSTM_φ(h_t)
z_t      ~ N(μ_φ, σ_φ)         ← reparameterization trick

# Learned Prior
h_{t-1}  = Enc(x_{t-1})
μ_ψ, σ_ψ = LSTM_ψ(h_{t-1})

# Prediction
g_t      = LSTM_θ(h_{t-1}, z_t)
x̂_t     = Dec(g_t)

# Loss
L2   = ||x̂_t - x_t||²
KL   = β · D_KL( N(μ_φ,σ_φ) || N(μ_ψ,σ_ψ) )
```

---

## ELBO 推導

從 VAE 的 variational lower bound 出發：

$$\log p_\theta(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})} \log p_\theta(\mathbf{x}|\mathbf{z}) - D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**步驟 1：分解 likelihood 到每個時間步。**
因為 frame predictor 是 autoregressive 的（$\mathbf{x}_t$ 只依賴 $\mathbf{x}_{1:t-1}$ 和 $\mathbf{z}_{1:t}$，未來的 $z$ 不影響現在）：

$$\log p_\theta(\mathbf{x}|\mathbf{z}) = \sum_t \log p_\theta(\mathbf{x}_t \mid \mathbf{x}_{1:t-1}, \mathbf{z}_{1:t})$$

**步驟 2：分解 KL 到每個時間步。**
Inference network 對各時間步獨立輸出後驗，因此：

$$q_\phi(\mathbf{z}|\mathbf{x}) = \prod_t q_\phi(\mathbf{z}_t|\mathbf{x}_{1:t})$$

KL 的乘積結構讓對數拆開後，交叉項對其他時間步積分都等於 1，化簡成：

$$D_{KL}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})) = \sum_t D_{KL}(q_\phi(\mathbf{z}_t|\mathbf{x}_{1:t}) \| p(\mathbf{z}_t))$$

**最終 ELBO（SVG-LP）：**

$$\mathcal{L} = \sum_{t=1}^{T} \Big[ \underbrace{\mathbb{E}_{q_\phi} \log p_\theta(\mathbf{x}_t \mid \mathbf{x}_{1:t-1}, \mathbf{z}_{1:t})}_{\text{= L2 重建 loss（} p_\theta \text{ 是固定方差高斯）}} - \underbrace{\beta\, D_{KL}(q_\phi(\mathbf{z}_t|\mathbf{x}_{1:t}) \| p_\psi(\mathbf{z}_t|\mathbf{x}_{1:t-1}))}_{\text{KL：拉近 posterior 與 learned prior}} \Big]$$

**β 的影響：**

| β | 效果 |
|---|------|
| 太小 | Inference net 自由複製 $x_t$，訓練好但測試時 prior ≠ posterior → 崩潰 |
| 太大 | 模型乾脆忽略 $z_t$，退化為確定性預測器 |
| 適中 | $z_t$ 只攜帶 prior 無法預測的隨機資訊 |

---

## 實驗

### 資料集

| 資料集 | 內容 | 挑戰 |
|--------|------|------|
| **SM-MNIST** | 64×64，MNIST 數字彈跳，碰牆時方向隨機 | 確定性運動 + 離散隨機事件 |
| **KTH Actions** | 真實人體動作（走、跑、拳擊等） | 關節位置輕微不確定性 |
| **BAIR Robot** | Sawyer 機械臂推物件 | 高度隨機動作 |

### 定量結果（SSIM↑）

> [!info]+ Figure 7：SVG-FP、SVG-LP 與 Deterministic 在 SM-MNIST（左）和 KTH（右）的 SSIM
> ![[image/papers/svg-2018/fig-p7-1.png]]
>
> SM-MNIST：SVG-LP（粉）>> SVG-FP（藍）>> Deterministic（綠虛線）。碰牆後確定性模型立即崩潰，而 SVG-LP 能維持高 SSIM 到更遠的時間步。KTH 上兩種 SVG 表現相近，因為動作本身規律性高。

> [!info]+ Figure 8：BAIR 資料集上與 Babaeizadeh et al. (2017) 的定量比較
> ![[image/papers/svg-2018/fig-p7-2.png]]
>
> SVG-FP 和 SVG-LP 在 SSIM 上均優於 Babaeizadeh et al.。PSNR 在後期步驟稍遜，作者指出 PSNR 偏好模糊（overly smooth）預測，可能是導致差距的原因。

### Learned Prior 的行為分析

> [!info]+ Figure 6：SVG-LP 的先驗方差隨時間步的變化
> ![[image/papers/svg-2018/fig-p6-4.png]]
>
> 500 條同步軌跡的 $\sigma_\psi(x_{1:t-1})$ 均值。**垂直線是數字碰牆的真實時刻（紅/藍代表兩顆數字）**。先驗方差的尖峰幾乎完美對齊碰牆瞬間——模型學會了什麼時候世界是隨機的。

### 與 Babaeizadeh et al. (2017) 的關鍵差異

| 項目 | Babaeizadeh et al. | SVG（本文） |
|------|-------------------|-------------|
| Inference network | 看整段影片 $q(z\|x_{1:T})$ | 逐步 $q_\phi(z_t\|x_{1:t})$ |
| 時間步之間的 latent | 同一分布採樣（time-variant）| 每步獨立後驗，更靈活 |
| 訓練方式 | 三階段（先 deterministic → 加 KL → 全 loss）| **單階段 end-to-end** |
| 生成品質 | 偏模糊 | **更清晰（sharper）** |

---

## 架構細節

```
Encoder：
  SM-MNIST → DCGAN discriminator，|h| = 128
  KTH/BAIR → VGG16（到最後 pooling 層），|h| = 128

Decoder：
  Encoder 的鏡像，pooling 換 upsampling，sigmoid 輸出

LSTM_θ (Prediction)：2 層，256 cells
LSTM_φ (Inference)： 1 層，256 cells
LSTM_ψ (Prior)：     1 層，256 cells

Latent dim：SM-MNIST → 10，KTH → 32，BAIR → 64
Optimizer：ADAM，lr = 0.002
β：1e-4（KTH/BAIR），1e-6（SM-MNIST）
```

---

## 與 Lab 4 CVAE 的連結

SVG 是**序列版的 CVAE**：

| CVAE（Lab 4） | SVG |
|--------------|-----|
| 條件 $c$（past frames）引導生成 | $x_{1:t-1}$ 作為 prediction model 的條件 |
| 靜態 latent $z$ | 每個時間步的 $z_t$ |
| Fixed prior $\mathcal{N}(0,I)$ | Learned prior $p_\psi(z_t\|x_{1:t-1})$ |
| Encoder = inference net | 測試時換成 prior 採樣 |

Lab 4 實作的核心就是這套 VAE training loop：encode → reparameterize → decode → L2 + KL loss。

---

## 思考題

1. **為什麼 inference network 在測試時不使用？**
   → 因為測試時沒有 ground truth $x_t$ 可以 encode。必須換成 prior（$\mathcal{N}(0,I)$ 或 $p_\psi$）來採樣 $z_t$。

2. **SVG-FP 和 SVG-LP 在 KTH 上幾乎一樣好，為什麼？**
   → KTH 動作規律，不確定性低且平穩分布在所有時間步，fixed prior 的固定雜訊恰好夠用；SM-MNIST 有離散的碰牆事件，才顯示出 learned prior 的優勢。

3. **β 太小會有什麼問題（與 β-VAE 類比）？**
   → Posterior 不受 prior 約束 → inference net 可以直接 encode $x_t$ 的全部資訊 → 訓練時重建完美，但測試時從 prior 採樣出的 $z_t$ 和 inference 的 $z_t$ 分布差距很大 → 生成崩潰（train-test mismatch）。
