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

    def forward(self, x243, x241):
        x244=operator.add(x243, (4, 64))
        x245=x241.view(x244)
        return x245

m = M().eval()
x243 = (1, 384, )
x241 = torch.randn(torch.Size([1, 384, 256]))
start = time.time()
output = m(x243, x241)
end = time.time()
print(end-start)
