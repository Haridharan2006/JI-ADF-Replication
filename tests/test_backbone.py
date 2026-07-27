import torch

from models.backbones.efficientnetv2 import ImageEncoder

model = ImageEncoder()

clinical = torch.randn(2,3,384,384)
derm = torch.randn(2,3,384,384)

output = model(derm, clinical)

print(output.shape)