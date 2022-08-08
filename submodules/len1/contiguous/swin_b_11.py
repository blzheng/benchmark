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

    def forward(self, x278):
        x279=x278.contiguous()
        return x279

m = M().eval()
x278 = torch.randn(torch.Size([16, 49, 49]))
start = time.time()
output = m(x278)
end = time.time()
print(end-start)