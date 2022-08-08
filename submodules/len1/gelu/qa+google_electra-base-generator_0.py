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

    def forward(self, x66):
        x67=torch._C._nn.gelu(x66)
        return x67

m = M().eval()
x66 = torch.randn(torch.Size([1, 384, 1024]))
start = time.time()
output = m(x66)
end = time.time()
print(end-start)
