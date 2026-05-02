---
tags: [DLP, lab5, RL, TODO]
created: 2026-04-25
deadline: 2026-05-05
---

# Lab 5 TODO — Value-Based Reinforcement Learning

> **Due: 2026/05/05 (Tue) 23:59**
> Late until 05/21 → 70% score | After 05/21 → 0分

---

## ✅ Task 1：Vanilla DQN on CartPole (15%)

- [ ] 用 `dqn.py` 為起點實作 Vanilla DQN
- [ ] 建立 fully connected 神經網路近似 Q-function
- [ ] 實作 **epsilon-greedy** 策略做 action selection
- [ ] 實作 **Experience Replay**（uniform sampling）+ **Target Network**
- [ ] 用 **Weight & Bias (WandB)** 記錄 episodic reward vs environment steps 曲線
- [ ] 儲存最佳 snapshot → `LAB5_{StudentID}_task1.pt`

---

## ✅ Task 2：Vanilla DQN on Atari Pong (20%)

- [ ] 預處理輸入 frames（灰階、resize、stack frames）
- [ ] 使用 **CNN** 作為 Q-function approximator
- [ ] 使用 `Pong-v5` 環境（ALE/Atari）
- [ ] 評估並用 WandB 繪製 total episodic reward vs environment steps
- [ ] 目標：平均 evaluation score ≥ 19（over 20 episodes）
- [ ] 儲存最佳 snapshot → `LAB5_{StudentID}_task2.pt`

> 🧮 分數公式：`(min(avg_score, 19) + 21) / 40 × 20%`
> ⚠️ Pong 訓練約需 ~1M steps，RTX 3090 約 20 小時，**要早點開始！**

---

## ✅ Task 3：Enhanced DQN (20% snapshots)

- [ ] 整合 **Double DQN (DDQN)**
- [ ] 整合 **Prioritized Experience Replay (PER)**
- [ ] 整合 **Multi-Step Return**
- [ ] 在 `Pong-v5` 比較 Enhanced vs Vanilla DQN 訓練效率
- [ ] 記錄「第一次穩定達到 score 19 的 timestep」
- [ ] 儲存 5 個 snapshots（在指定步數）：

| Snapshot 檔名 | Steps | 滿分條件 |
|---|---|---|
| `LAB5_{ID}_task3_600000.pt` | 600k | 20% |
| `LAB5_{ID}_task3_1000000.pt` | 1M | 17% |
| `LAB5_{ID}_task3_1500000.pt` | 1.5M | 15% |
| `LAB5_{ID}_task3_2000000.pt` | 2M | 13% |
| `LAB5_{ID}_task3_2500000.pt` | 2.5M | 11% |
| `LAB5_{ID}_task3_best.pt` | 任意達到 score 19 | — |

---

## ✅ Report (60%)

- [ ] **Introduction (5%)**：高層次介紹，提及最重要發現與報告結構
- [ ] **Implementation (20%)**：解釋概念 + 實作細節，涵蓋：
  - [ ] DQN 的 Bellman error 如何計算？
  - [ ] 如何將 DQN 改成 Double DQN？
  - [ ] PER 的 memory buffer 如何實作？
  - [ ] 如何將 1-step return 改成 multi-step return？
  - [ ] 如何用 WandB 追蹤模型效能？
- [ ] **Analysis & Discussion (25%)**：
  - [ ] Task 1, 2, 3 各自的訓練曲線（x 軸 = environment steps）
  - [ ] 有/無 DQN enhancements 的 sample efficiency 分析
  - [ ] 每個 technique 的 ablation study（各別測試）
- [ ] **Bonus (up to 10%)**：其他訓練策略分析
- [ ] 每個 task 的 evaluation **截圖**（缺一張 -3 分）
- [ ] 包含可重現結果的**執行指令**（缺指令 -5 分）

---

## ✅ Demo Video（5~6 分鐘，英文）

- [ ] Source code 介紹 (~2 min)：描述實作方式
- [ ] Model performance demo (~3 min)：展示 Task 1, 2, 3 模型
- [ ] 存成 `LAB5_{StudentID}.mp4`
- [ ] ⚠️ 沒有 demo video → model snapshots **不計分**

---

## ✅ 提交打包 (`LAB5_{StudentID}.zip`)

```
LAB5_StudentID.zip
├── LAB5_StudentID_Code/
│   ├── dqn.py
│   ├── requirements.txt
│   └── (其他 .py / .sh)
├── LAB5_StudentID.pdf          ← 報告
├── LAB5_StudentID.mp4          ← Demo video
├── LAB5_StudentID_task1.pt
├── LAB5_StudentID_task2.pt
├── LAB5_StudentID_task3_600000.pt
├── LAB5_StudentID_task3_1000000.pt
├── LAB5_StudentID_task3_1500000.pt
├── LAB5_StudentID_task3_2000000.pt
├── LAB5_StudentID_task3_2500000.pt
└── LAB5_StudentID_task3_best.pt
```

- [ ] 確認所有檔名正確（錯一個 -5 分）
- [ ] 加入 `requirements.txt`
- [ ] 附上 evaluation 執行指令（例：`python test_model.py --model_path ...`）
- [ ] evaluation seeds: 0 ~ 19

---

## 環境需求

```
Python >= 3.8
gymnasium == 1.1.1
ale-py >= 0.10.0
opencv-python
torch
wandb
```

---

## 不能改的 Class 結構（`dqn.py`）

- `DQN`
- `PrioritizedReplayBuffer`
- `AtariPreprocessor`
- `DQNAgent`
