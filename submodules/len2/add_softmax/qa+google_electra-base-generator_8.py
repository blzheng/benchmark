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

    def forward(self, x386):
        x388=operator.add(x386, self._tensor_constant20)
        x389=torch.nn.functional.softmax(x388,dim=-1, _stacklevel=3, dtype=None)
        return x389

m = M().eval()
x386 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x386)
end = time.time()
print(end-start)
