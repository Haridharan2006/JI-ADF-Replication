import torch

from models.adf import AdaptiveDecisionFusion

model = AdaptiveDecisionFusion()

image = torch.randn(2,11)
metadata = torch.randn(2,11)
joint = torch.randn(2,11)

logits, probs, weights = model(
    image,
    metadata,
    joint
)

print(logits.shape)
print(probs.shape)
print(weights.shape)