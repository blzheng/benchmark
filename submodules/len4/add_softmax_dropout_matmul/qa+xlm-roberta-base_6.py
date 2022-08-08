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
        self.dropout19 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x301, x293):
        x303=operator.add(x301, self._tensor_constant20)
        x304=torch.nn.functional.softmax(x303,dim=-1, _stacklevel=3, dtype=None)
        x305=self.dropout19(x304)
        x306=torch.matmul(x305, x293)
        return x306

m = M().eval()
x301 = torch.randn(torch.Size([1, 12, 384, 384]))
x293 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x301, x293)
end = time.time()
print(end-start)
