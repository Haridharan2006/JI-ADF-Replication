import torch

from models.heads.classifier_heads import ClassifierHeads

model = ClassifierHeads()

image = torch.randn(2,1280)
metadata = torch.randn(2,256)
joint = torch.randn(2,512)

img, meta, joint_logits = model(
    image,
    metadata,
    joint
)

print(img.shape)
print(meta.shape)
print(joint_logits.shape)