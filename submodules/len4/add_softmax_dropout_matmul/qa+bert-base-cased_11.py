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
        self.dropout34 = Dropout(p=0.1, inplace=False)
        self._tensor_constant150110 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x511, x503):
        x513=operator.add(x511, self._tensor_constant150110)
        x514=torch.nn.functional.softmax(x513,dim=-1, _stacklevel=3, dtype=None)
        x515=self.dropout34(x514)
        x516=torch.matmul(x515, x503)
        return x516

m = M().eval()
x511 = torch.randn(torch.Size([1, 12, 384, 384]))
x503 = torch.randn(torch.Size([1, 12, 384, 64]))
start = time.time()
output = m(x511, x503)
end = time.time()
print(end-start)
