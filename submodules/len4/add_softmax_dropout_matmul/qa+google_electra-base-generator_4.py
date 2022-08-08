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

    def forward(self, x218, x210):
        x220=operator.add(x218, self._tensor_constant20)
        x221=torch.nn.functional.softmax(x220,dim=-1, _stacklevel=3, dtype=None)
        x222=self.dropout13(x221)
        x223=torch.matmul(x222, x210)
        return x223

m = M().eval()
x218 = torch.randn(torch.Size([1, 4, 384, 384]))
x210 = torch.randn(torch.Size([1, 4, 384, 64]))
start = time.time()
output = m(x218, x210)
end = time.time()
print(end-start)
