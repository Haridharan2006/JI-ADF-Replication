import torch

from models.ji_adf import JIADFModel

model = JIADFModel(metadata_input_dim=14)

clinical = torch.randn(2,3,384,384)
derm = torch.randn(2,3,384,384)
metadata = torch.randn(2,14)

output = model(
    derm,
    clinical,
    metadata
)

for key,value in output.items():

    if torch.is_tensor(value):

        print(key, value.shape)