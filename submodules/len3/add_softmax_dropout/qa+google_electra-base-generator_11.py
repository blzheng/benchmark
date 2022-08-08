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
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x512):
        x514=operator.add(x512, self._tensor_constant20)
        x515=torch.nn.functional.softmax(x514,dim=-1, _stacklevel=3, dtype=None)
        x516=self.dropout34(x515)
        return x516

m = M().eval()
x512 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x512)
end = time.time()
print(end-start)
