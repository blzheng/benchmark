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
        self.dropout4 = Dropout(p=0.1, inplace=False)
        self._tensor_constant20 = torch.rand(torch.Size([1, 1, 1, 384])).to(torch.float32)

    def forward(self, x92):
        x94=operator.add(x92, self._tensor_constant20)
        x95=torch.nn.functional.softmax(x94,dim=-1, _stacklevel=3, dtype=None)
        x96=self.dropout4(x95)
        return x96

m = M().eval()
x92 = torch.randn(torch.Size([1, 4, 384, 384]))
start = time.time()
output = m(x92)
end = time.time()
print(end-start)
