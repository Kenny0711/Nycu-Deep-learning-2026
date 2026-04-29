# NYCU Deep Learning Practice 2026 Spring

**Student:** 楊正豪 (314553044)

---

## Table of Contents

- [Lab 1 — Backpropagation](#lab-1--backpropagation)
- [Lab 2 — Binary Semantic Segmentation](#lab-2--binary-semantic-segmentation)
- [Lab 3 — MaskGIT for Image Inpainting](#lab-3--maskgit-for-image-inpainting)
- [Lab 4 — Conditional VAE for Video Prediction](#lab-4--conditional-vae-for-video-prediction)
- [Lab 5 — Deep Q-Network (DQN)](#lab-5--deep-q-network-dqn)
- [Final Project — SUMO Traffic Simulation with LCPO](#final-project--sumo-traffic-simulation-with-lcpo)

---

## Lab 1 — Backpropagation

**Goal:** Build and train a fully-connected neural network from scratch using only NumPy, and verify it on XOR and Linear classification tasks.

### Framework
- Pure Python + NumPy (no deep learning framework)

### What was done
- Built a 3-layer MLP with forward pass and manual backpropagation
- Implemented Sigmoid, ReLU, and linear activation functions
- Implemented SGD and SGD with Momentum optimizers
- Compared model performance across different activations and optimizers on XOR and Linear datasets

### Key Files

| File | Description |
|------|-------------|
| `model_question.py` | Core MLP: forward, backpropagation, weight update |
| `data.py` | Data generators for Linear and XOR datasets |
| `main.py` | Training loop and result visualization |
| `config.json` | Hyperparameter configuration |

---

## Lab 2 — Binary Semantic Segmentation

**Goal:** Train UNet and ResNet34-UNet to perform binary semantic segmentation on the Oxford-IIIT Pet dataset.

### Framework
- PyTorch

### What was done
- Implemented UNet and ResNet34-UNet architectures
- Trained on Oxford-IIIT Pet dataset for foreground/background pet segmentation
- Evaluated with Dice score; generated Kaggle submission CSV

### Key Files

| File | Description |
|------|-------------|
| `src/model/unet.py` | UNet architecture with Conv blocks and transposed convolution upsampling |
| `src/model/resnet34_unet.py` | ResNet34 encoder + UNet decoder |
| `src/train.py` | Training loop with configurable model, epochs, lr, batch size |
| `src/evaluate.py` | Dice score evaluation |
| `src/oxford_pet.py` | Oxford-IIIT Pet dataset loader |
| `src/inference.py` | Inference and CSV generation for Kaggle |

---

## Lab 3 — MaskGIT for Image Inpainting

**Goal:** Use a pretrained VQGAN with a Masked Bidirectional Transformer (MaskGIT) to iteratively reconstruct masked regions in images.

### Framework
- PyTorch

### What was done
- Trained a Masked Generative Image Transformer (MaskGIT) on top of a frozen VQGAN codebook
- Implemented mask scheduling strategies (cosine, etc.) for iterative token prediction
- Performed image inpainting inference and evaluated with FID score

### Key Files

| File | Description |
|------|-------------|
| `models/VQGAN_Transformer.py` | Combined VQGAN + Masked Transformer model |
| `models/Transformer/transformer.py` | Bidirectional transformer for masked token prediction |
| `models/VQGAN/VQGAN.py` | Pretrained VQGAN encoder/decoder/codebook |
| `training_transformer.py` | Transformer training loop with mask scheduling |
| `inpainting.py` | Iterative masked inpainting inference |
| `utils.py` | Data loading utilities |

---

## Lab 4 — Conditional VAE for Video Prediction

**Goal:** Implement SVG-LP (Stochastic Video Generation with Learned Prior) to predict future dance video frames using a Conditional VAE with learned prior and teacher forcing.

### Framework
- PyTorch

### What was done
- Implemented VAE reparameterization trick in `Gaussian_Predictor`
- Implemented KL annealing strategies (Cyclical, Monotonic, None) to stabilize training
- Implemented training step with MSE reconstruction loss + KL divergence
- Implemented teacher forcing ratio decay schedule
- Evaluated with PSNR metric on 630-frame video sequences

### Key Files

| File | Description |
|------|-------------|
| `Lab4_template/Trainer.py` | Training loop with KL annealing and teacher forcing |
| `Lab4_template/Tester.py` | Test inference generating 630-frame sequences |
| `Lab4_template/dataloader.py` | Dance video dataset loader (frames + pose labels) |
| `Lab4_template/modules/modules.py` | Gaussian_Predictor, Decoder_Fusion, kl_annealing |
| `Lab4_template/modules/layers.py` | Encoder/Decoder layer implementations |

---

## Lab 5 — Deep Q-Network (DQN)

**Goal:** Implement DQN and its variants to solve discrete control tasks.

### Framework
- PyTorch

### What was done
- Implemented DQN with experience replay and target network
- Extended to multiple task variants (Task 2, Task 3)
- Evaluated agent performance with recorded evaluation videos

### Key Files

| File | Description |
|------|-------------|
| `dqn.py` | Main DQN implementation |
| `dqn_task2.py` | DQN variant for Task 2 |
| `dqn_task3.py` | DQN variant for Task 3 |
| `test_model.py` | Model evaluation and video recording |
| `eval_videos/` | Recorded evaluation episodes (MP4) |

---

## Final Project — SUMO Traffic Simulation with LCPO

**Goal:** Apply reinforcement learning to traffic signal control using SUMO (Simulation of Urban MObility) with a Taipei road network.

### Framework
- Python + SUMO

### What was done
- Built a SUMO traffic simulation environment for Taipei road topology
- Implemented traffic signal control with RL agents
- Crawled real Taipei road data via TDX API for network construction

### Key Files

| File | Description |
|------|-------------|
| `final/sumo_taipei/sumo_scene_builder.py` | Builds SUMO network from Taipei road data |
| `final/sumo_taipei/run_simulation.py` | Runs traffic simulation with RL agent |
| `final/sumo_taipei/tdx_crawler.py` | Crawls Taipei road data from TDX API |
| `final/sumo_taipei/nets/` | SUMO network config files (cross, taipei route) |
