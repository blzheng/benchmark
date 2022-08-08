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
        self.dropout10 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x176, x168):
        x178=operator.add(x176, self._tensor_constant20)
        x179=torch.nn.functional.softmax(x178,dim=-1, _stacklevel=3, dtype=None)
        x180=self.dropout10(x179)
        x181=torch.matmul(x180, x168)
        return x181

m = M().eval()
x176 = torch.randn(torch.Size([1, 4, 384, 384]))
x168 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x176, x168)
end = time.time()
print(end-start)
