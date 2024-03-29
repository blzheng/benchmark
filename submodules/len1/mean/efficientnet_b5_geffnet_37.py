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

    def forward(self, x551):
        x552=x551.mean((2, 3),keepdim=True)
        return x552

m = M().eval()
x551 = torch.randn(torch.Size([1, 3072, 7, 7]))
start = time.time()
output = m(x551)
end = time.time()
print(end-start)
