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
        self.sigmoid17 = Sigmoid()

    def forward(self, x291, x287):
        x292=self.sigmoid17(x291)
        x293=operator.mul(x292, x287)
        return x293

m = M().eval()
x291 = torch.randn(torch.Size([1, 576, 1, 1]))
x287 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x291, x287)
end = time.time()
print(end-start)
