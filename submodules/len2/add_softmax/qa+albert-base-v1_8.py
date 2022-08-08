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

    def forward(self, x346):
        x348=operator.add(x346, self._tensor_constant20)
        x349=torch.nn.functional.softmax(x348,dim=-1, _stacklevel=3, dtype=None)
        return x349

m = M().eval()
x346 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x346)
end = time.time()
print(end-start)
