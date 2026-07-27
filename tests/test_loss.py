import torch

from losses.loss import JIADFLoss

criterion = JIADFLoss()

output = {

    "image_logits":torch.randn(4,11),

    "metadata_logits":torch.randn(4,11),

    "joint_logits":torch.randn(4,11),

    "prediction_logits":torch.randn(4,11)

}

labels = torch.randint(0,11,(4,))

loss = criterion(output,labels)

print(loss)