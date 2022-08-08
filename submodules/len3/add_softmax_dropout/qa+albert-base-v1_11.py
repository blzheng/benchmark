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

    def forward(self, x457):
        x459=operator.add(x457, self._tensor_constant20)
        x460=torch.nn.functional.softmax(x459,dim=-1, _stacklevel=3, dtype=None)
        x461=self.dropout1(x460)
        return x461

m = M().eval()
x457 = torch.randn(torch.Size([1, 12, 384, 384]))
start = time.time()
output = m(x457)
end = time.time()
print(end-start)
