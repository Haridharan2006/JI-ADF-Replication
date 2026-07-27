from utils.seed import set_seed

import torch

set_seed(42)

a=torch.rand(5)

set_seed(42)

b=torch.rand(5)

print(a)

print(b)

print(torch.equal(a,b))