import torch

from models.metadata_encoder import MetadataEncoder

model = MetadataEncoder(input_dim=14)

x = torch.randn(2,14)

output = model(x)

print(output.shape)