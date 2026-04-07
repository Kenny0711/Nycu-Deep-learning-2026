# NYCU Deep Learning Practice 2026 Spring

Student: 楊正豪 (314553044)

---

## Lab 1 — Backpropagation

**Core idea:** Implement a fully-connected neural network from scratch using only NumPy, and verify it on XOR and Linear classification tasks.

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
| `main.py` | Training loop, result visualization |
| `config.json` | Hyperparameter configuration |

---

## Lab 2 — Binary Semantic Segmentation

**Core idea:** Train UNet and ResNet34-UNet to perform binary semantic segmentation on the Oxford-IIIT Pet dataset.

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

**Core idea:** Use a pretrained VQGAN with a Masked Bidirectional Transformer (MaskGIT) to iteratively reconstruct masked regions in images.

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
