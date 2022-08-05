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

    def forward(self, x48):
        x49=x48.mean((2, 3),keepdim=True)
        return x49

m = M().eval()
x48 = torch.randn(torch.Size([1, 192, 56, 56]))
start = time.time()
output = m(x48)
end = time.time()
print(end-start)
