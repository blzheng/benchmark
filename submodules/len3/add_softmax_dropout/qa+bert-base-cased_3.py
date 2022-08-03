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
        self._tensor_constant70070 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x175):
        x177=operator.add(x175, self._tensor_constant70070)
        x178=torch.nn.functional.softmax(x177,dim=-1, _stacklevel=3, dtype=None)
        x179=self.dropout10(x178)
        return x179

m = M().eval()
x175 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x175)
end = time.time()
print(end-start)
