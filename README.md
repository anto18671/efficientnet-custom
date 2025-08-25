````markdown
# EfficientNet-Custom (PyTorch)

A custom PyTorch implementation of **EfficientNet**, including **Squeeze-and-Excitation (SE)** blocks and **Mobile Inverted Bottleneck (MBConv)** blocks.  
This project is designed to provide a clear and well-documented backbone implementation for experimentation, learning, and extension.

---

## ✨ Features

- **Squeeze-and-Excitation (SE) blocks** for channel-wise attention.
- **MBConv blocks** with optional expansion and residual connections.
- **Flexible configuration** with width/depth multipliers.
- **Customizable EfficientNet backbone** with clean modular code.
- **Standardized weight initialization** for stability.
- **Dropout & BatchNorm** for better regularization.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/anto18671/efficientnet-custom.git
cd efficientnet-custom
```
````

Install dependencies:

```bash
pip install - r requirements.txt
```

---

## 🚀 Usage

### Import the model

```python
from efficientnet import EfficientNet

# Example base configuration (t, c, n, s):
# t = expansion ratio
# c = output channels
# n = number of repeats
# s = stride
base_cfg = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 40, 2, 2),
    (6, 80, 3, 2),
    (6, 112, 3, 1),
    (6, 192, 4, 2),
    (6, 320, 1, 1),
]

# Build model
model = EfficientNet(base_cfg=base_cfg, num_classes=1000)

# Example forward pass
import torch
x = torch.randn(1, 3, 224, 224)
y = model(x)
print(y.shape)  # torch.Size([1, 1000])
```

---

## 🧩 Code Structure

```
efficientnet-custom/
│── efficientnet.py   # Model implementation
│── README.md         # Documentation
```

- **SqueezeExcite**: Channel recalibration via global average pooling and gating.
- **MBConv**: Mobile inverted bottleneck with depthwise separable convolutions + SE.
- **EfficientNet**: Backbone builder with configurable blocks, stem, and classifier.

---

## 🔬 Example Training Loop

```python
import torch.optim as optim
import torch.nn as nn

# Initialize model
model = EfficientNet(base_cfg=base_cfg, num_classes=10)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Dummy batch
x = torch.randn(8, 3, 224, 224)
labels = torch.randint(0, 10, (8,))

# Forward + backward
outputs = model(x)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

---

## 📖 References

- [EfficientNet: Rethinking Model Scaling for CNNs (ICML 2019)](https://arxiv.org/abs/1905.11946)
- [Squeeze-and-Excitation Networks (CVPR 2018)](https://arxiv.org/abs/1709.01507)

---

## 📜 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for details.
