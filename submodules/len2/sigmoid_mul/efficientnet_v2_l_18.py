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
        self.sigmoid18 = Sigmoid()

    def forward(self, x411, x407):
        x412=self.sigmoid18(x411)
        x413=operator.mul(x412, x407)
        return x413

m = M().eval()
x411 = torch.randn(torch.Size([1, 1344, 1, 1]))
x407 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x411, x407)
end = time.time()
print(end-start)
