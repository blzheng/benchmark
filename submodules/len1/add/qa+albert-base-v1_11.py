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
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x87):
        x89=operator.add(x87, self._tensor_constant20)
        return x89

m = M().eval()
x87 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
