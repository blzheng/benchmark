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
        self.sigmoid4 = Sigmoid()

    def forward(self, x156, x152):
        x157=self.sigmoid4(x156)
        x158=operator.mul(x157, x152)
        return x158

m = M().eval()
x156 = torch.randn(torch.Size([1, 640, 1, 1]))
x152 = torch.randn(torch.Size([1, 640, 14, 14]))
start = time.time()
output = m(x156, x152)
end = time.time()
print(end-start)
