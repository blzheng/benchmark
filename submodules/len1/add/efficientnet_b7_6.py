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

    def forward(self, x130, x115):
        x131=operator.add(x130, x115)
        return x131

m = M().eval()
x130 = torch.randn(torch.Size([1, 48, 56, 56]))
x115 = torch.randn(torch.Size([1, 48, 56, 56]))
start = time.time()
output = m(x130, x115)
end = time.time()
print(end-start)
