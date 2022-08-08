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
        self.dropout31 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x469, x461):
        x471=operator.add(x469, self._tensor_constant20)
        x472=torch.nn.functional.softmax(x471,dim=-1, _stacklevel=3, dtype=None)
        x473=self.dropout31(x472)
        x474=torch.matmul(x473, x461)
        return x474

m = M().eval()
x469 = torch.randn(torch.Size([1, 12, 384, 384]))
x461 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x469, x461)
end = time.time()
print(end-start)
