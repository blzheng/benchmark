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

    def forward(self, x161, x158):
        x163=operator.add(x161, self._tensor_constant20)
        x164=torch.nn.functional.softmax(x163,dim=-1, _stacklevel=3, dtype=None)
        x165=self.dropout1(x164)
        x166=torch.matmul(x165, x158)
        return x166

m = M().eval()
x161 = torch.randn(torch.Size([1, 12, 384, 384]))
x158 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x161, x158)
end = time.time()
print(end-start)
