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

    def forward(self, x87):
        x89=operator.add(x87, self._tensor_constant20)
        x90=torch.nn.functional.softmax(x89,dim=-1, _stacklevel=3, dtype=None)
        x91=self.dropout1(x90)
        return x91

m = M().eval()
x87 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)
