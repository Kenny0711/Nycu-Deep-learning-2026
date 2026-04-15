# Lab 4 - Conditional VAE for Video Prediction (CVAE / SVG-LP)

## 概覽

本 Lab 要實作一個基於 **SVG-LP (Stochastic Video Generation with a Learned Prior)** 的舞蹈影片預測模型。  
給定第一幀 + 630 幀 pose label 序列，模型需預測出對應的 630 幀影片。  
最終提交 `submission.csv` 到 Kaggle，評分指標為 **PSNR**（越高越好）。

---

## 模型架構

```
RGB frame  ──► RGB_Encoder ──► frame feature (F_dim=128)
                                        │
Pose label ──► Label_Encoder ──► label feature (L_dim=32)
                                        │
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
              Gaussian_Predictor               Decoder_Fusion
              (posterior q(z|x,l))         (fuses frame+label+z)
              outputs: z, mu, logvar               │
                                                   ▼
                                              Generator
                                         (predicts next frame)
```

---

## 需要實作的部分（全部標記 `# TODO` 或 `raise NotImplementedError`）

### 1. `modules/modules.py` — `Gaussian_Predictor.reparameterize()`

**位置**: `modules/modules.py:79`

實作 VAE 的 **reparameterization trick**：

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

---

### 2. `Trainer.py` — `kl_annealing` class

**位置**: `Trainer.py:36-52`

KL annealing 用來控制訓練過程中 KL loss 的權重 beta，防止 posterior collapse。

需實作：

#### (a) `frange_cycle_linear(n_iter, start, stop, n_cycle, ratio)`
產生 cyclical linear annealing 的 beta schedule（一個 list）：
- 每個 cycle 分成兩段：線性上升（佔 ratio 比例）和固定在 stop
- 參考論文: *Cyclical Annealing Schedule* (Fu et al., 2019)

```python
def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and int(i + c * period) < n_iter:
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L
```

#### (b) `__init__(self, args, current_epoch=0)`
根據 `args.kl_anneal_type` 初始化：
- `'Cyclical'`：用 `frange_cycle_linear` 產生整個 schedule
- `'Monotonic'`：線性從 0 增長到 1（n_cycle=1, ratio=1）
- `'None'`：beta 固定為 1.0

#### (c) `update()`
每個 epoch 結束後呼叫，推進 schedule index

#### (d) `get_beta()`
回傳當前 beta 值

---

### 3. `Trainer.py` — `VAE_Model.training_one_step()`

**位置**: `Trainer.py:124`

單一 batch 的訓練步驟：

```
輸入: img (B, T, C, H, W), label (B, T, C, H, W), adapt_TeacherForcing (bool)

流程:
1. permute → (T, B, C, H, W)
2. 用第 0 幀初始化 prev_frame
3. 對每一時間步 t = 1..T-1:
   a. 編碼 prev_frame → frame_feat
   b. 編碼 label[t]  → label_feat
   c. 編碼 img[t]    → gt_feat   (ground truth，訓練時才有)
   d. Gaussian_Predictor(gt_feat, label_feat) → z, mu, logvar
   e. Decoder_Fusion(frame_feat, label_feat, z) → fused
   f. Generator(fused) → pred_frame
   g. Teacher Forcing: 
      - ON  → prev_frame = img[t]  (用 GT)
      - OFF → prev_frame = pred_frame (自迴歸)
4. loss = MSE(pred, gt) + beta * KL(mu, logvar)
5. optim.zero_grad() → loss.backward() → optimizer_step()
6. return loss
```

---

### 4. `Trainer.py` — `VAE_Model.val_one_step()`

**位置**: `Trainer.py:128`

驗證步驟（不更新梯度，無 GT frame 輸入 Gaussian_Predictor）：
- 在推論時，從 **prior N(0,1)** 取樣 z（而非 posterior）
- 計算 PSNR 作為評估指標

```
流程:
1. permute → (T, B, C, H, W)
2. 用第 0 幀初始化 prev_frame
3. 對每一時間步 t:
   a. 編碼 prev_frame → frame_feat
   b. 編碼 label[t]  → label_feat
   c. 從 N(0,1) 取樣 z (shape 同 mu)
   d. Decoder_Fusion(frame_feat, label_feat, z) → fused
   e. Generator(fused) → pred_frame
   f. prev_frame = pred_frame
4. 計算與 GT 的 MSE / PSNR 回傳
```

---

### 5. `Trainer.py` — `VAE_Model.teacher_forcing_ratio_update()`

**位置**: `Trainer.py:171`

Teacher Forcing Ratio (tfr) 控制訓練時使用 GT frame 的機率：
- 從 `tfr_sde` 這個 epoch 開始，每 epoch 遞減 `tfr_d_step`
- 最低不低於 0

```python
def teacher_forcing_ratio_update(self):
    if self.current_epoch >= self.tfr_sde:
        self.tfr = max(0, self.tfr - self.tfr_d_step)
```

---

### 6. `Tester.py` — `Test_model.val_one_step()`

**位置**: `Tester.py:113`

測試推論，輸出 630 幀預測序列：
- 輸入只有第 0 幀影像 + 630 幀 pose label
- 自迴歸生成，從 prior N(0,1) 取樣 z
- 輸出 shape 必須為 `(1, 630, 3, 32, 64)`

```python
# 範例骨架
for t in range(label.shape[0]):  # 630 steps
    frame_feat = self.frame_transformation(prev_frame)
    label_feat = self.label_transformation(label[t])
    z = torch.randn(frame_feat.shape[0], self.args.N_dim, 
                    frame_feat.shape[2], frame_feat.shape[3]).to(device)
    fused = self.Decoder_Fusion(frame_feat, label_feat, z)
    pred = self.Generator(fused)
    decoded_frame_list.append(pred.cpu())
    label_list.append(label[t].cpu())
    prev_frame = pred
```

---

## 評估指標

- **PSNR** (Peak Signal-to-Noise Ratio)：越高越好
  - 公式：`PSNR = 20*log10(1.0) - 10*log10(MSE)`
  - 目標：驗證集 PSNR 達到 baseline 以上

---

## 訓練指令範例

```bash
# 訓練（Cyclical KL annealing）
python Trainer.py \
  --DR /path/to/LAB4_Dataset \
  --save_root ./checkpoints \
  --device cuda \
  --num_epoch 70 \
  --kl_anneal_type Cyclical \
  --kl_anneal_cycle 10 \
  --kl_anneal_ratio 1.0 \
  --tfr 1.0 \
  --tfr_sde 10 \
  --tfr_d_step 0.1 \
  --fast_train

# 測試並生成 submission.csv
python Tester.py \
  --DR /path/to/LAB4_Dataset \
  --save_root ./output \
  --ckpt_path ./checkpoints/epoch=XX.ckpt \
  --device cuda
```

---

## 超參數一覽

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 2 | 批次大小 |
| `lr` | 0.001 | 初始學習率 |
| `num_epoch` | 70 | 訓練總 epoch |
| `train_vi_len` | 16 | 訓練影片片段長度 |
| `val_vi_len` | 630 | 驗證影片長度 |
| `F_dim` | 128 | RGB frame feature 維度 |
| `L_dim` | 32 | Pose label feature 維度 |
| `N_dim` | 12 | Latent noise 維度 |
| `D_out_dim` | 192 | Decoder 輸出維度 |
| `tfr` | 1.0 | 初始 teacher forcing ratio |
| `tfr_sde` | 10 | TFR 開始衰減的 epoch |
| `tfr_d_step` | 0.1 | TFR 每 epoch 衰減量 |
| `kl_anneal_type` | Cyclical | KL annealing 類型 |
| `kl_anneal_cycle` | 10 | Cyclical annealing 週期數 |
| `kl_anneal_ratio` | 1.0 | 線性上升佔週期比例 |

---

## 資料集結構

```
LAB4_Dataset/
├── train/
│   ├── train_img/    # 訓練影像幀 (PNG)
│   └── train_label/  # 訓練 pose label (PNG)
├── val/
│   ├── val_img/
│   └── val_label/
└── test/
    ├── test_img/     # 只有第 0 幀
    └── test_label/   # 630 幀 pose label
```

---

## 需要提交的內容

1. **`submission.csv`** — 上傳到 Kaggle
2. **訓練/驗證 Loss 曲線圖**
3. **PSNR 曲線圖**
4. **生成的 GIF 動畫**（`pred_seq{idx}.gif`）
5. **實驗報告**：比較不同 KL annealing 策略（Cyclical vs Monotonic vs None）和 Teacher Forcing 設定的效果

---

## 論文重點 (SVG-LP)

- **paper.pdf**: *Stochastic Video Generation with a Learned Prior* (Denton & Fergus, 2018)
- 核心思想：訓練時用 posterior q(z|xt, x1:t-1) 提供資訊豐富的 z；測試時用 learned prior p(z|x1:t-1) 取樣
- 本 Lab 簡化版：prior 固定為 N(0,1)（即 SVG-FP 變體），但用 pose label 作為條件（Conditional）
- KL annealing 防止 posterior collapse，讓 z 學到有意義的資訊
