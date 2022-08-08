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
        self.dropout1 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x235, x232):
        x237=operator.add(x235, self._tensor_constant20)
        x238=torch.nn.functional.softmax(x237,dim=-1, _stacklevel=3, dtype=None)
        x239=self.dropout1(x238)
        x240=torch.matmul(x239, x232)
        return x240

m = M().eval()
x235 = torch.randn(torch.Size([1, 12, 384, 384]))
x232 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x235, x232)
end = time.time()
print(end-start)
