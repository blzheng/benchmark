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
        self.dropout25 = Dropout(p=0.1, inplace=False)
        self._tensor_constant70070 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x385):
        x387=operator.add(x385, self._tensor_constant70070)
        x388=torch.nn.functional.softmax(x387,dim=-1, _stacklevel=3, dtype=None)
        x389=self.dropout25(x388)
        return x389

m = M().eval()
x385 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x385)
end = time.time()
print(end-start)
