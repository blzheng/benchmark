import torch
from torch import tensor
import torch.nn as nn
from torch.nn import *
import torchvision
import torchvision.models as models
from torchvision.ops.stochastic_depth import stochastic_depth
import time
import builtins
import operator

class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()

    def forward(self, x494, x492):
        x495=operator.add(x494, (12, 64))
        x496=x492.view(x495)
        return x496

m = M().eval()
x494 = (1, 384, )
x492 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x494, x492)
end = time.time()
print(end-start)
