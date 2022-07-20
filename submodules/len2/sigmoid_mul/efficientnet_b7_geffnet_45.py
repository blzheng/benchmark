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

    def forward(self, x674, x670):
        x675=x674.sigmoid()
        x676=operator.mul(x670, x675)
        return x676

m = M().eval()
x674 = torch.randn(torch.Size([1, 2304, 1, 1]))
x670 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x674, x670)
end = time.time()
print(end-start)
