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
        self.dropout7 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x133):
        x135=operator.add(x133, self._tensor_constant20)
        x136=torch.nn.functional.softmax(x135,dim=-1, _stacklevel=3, dtype=None)
        x137=self.dropout7(x136)
        return x137

m = M().eval()
x133 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x133)
end = time.time()
print(end-start)
