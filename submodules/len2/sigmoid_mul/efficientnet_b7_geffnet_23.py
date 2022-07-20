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

    def forward(self, x346, x342):
        x347=x346.sigmoid()
        x348=operator.mul(x342, x347)
        return x348

m = M().eval()
x346 = torch.randn(torch.Size([1, 960, 1, 1]))
x342 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x346, x342)
end = time.time()
print(end-start)
