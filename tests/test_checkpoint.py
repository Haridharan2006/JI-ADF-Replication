import torch

from models.metadata_encoder import MetadataEncoder
from utils.checkpoint import save_checkpoint, load_checkpoint

model = MetadataEncoder(input_dim=14)

optimizer = torch.optim.Adam(model.parameters())

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=1,
    best_metric=0.9,
    filepath="temp.pth"
)

model, optimizer, epoch, best_metric = load_checkpoint(
    model=model,
    optimizer=optimizer,
    filepath="temp.pth"
)

print("Checkpoint OK")
print(epoch)
print(best_metric)