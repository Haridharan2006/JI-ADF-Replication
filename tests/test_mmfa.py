import torch

from models.fusion.mmfa import MMFA

model = MMFA()

image = torch.randn(2,1280)
metadata = torch.randn(2,256)

output = model(image, metadata)

print(output.shape)