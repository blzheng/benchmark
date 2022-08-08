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
        self.dropout13 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x217, x209):
        x219=operator.add(x217, self._tensor_constant20)
        x220=torch.nn.functional.softmax(x219,dim=-1, _stacklevel=3, dtype=None)
        x221=self.dropout13(x220)
        x222=torch.matmul(x221, x209)
        return x222

m = M().eval()
x217 = torch.randn(torch.Size([1, 12, 384, 384]))
x209 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x217, x209)
end = time.time()
print(end-start)
