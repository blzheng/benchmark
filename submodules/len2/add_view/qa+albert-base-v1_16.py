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

    def forward(self, x224, x216):
        x225=operator.add(x224, (12, 64))
        x226=x216.view(x225)
        return x226

m = M().eval()
x224 = (1, 384, )
x216 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x224, x216)
end = time.time()
print(end-start)
