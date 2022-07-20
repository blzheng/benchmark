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
        self.sigmoid23 = Sigmoid()

    def forward(self, x437, x433):
        x438=self.sigmoid23(x437)
        x439=operator.mul(x438, x433)
        return x439

m = M().eval()
x437 = torch.randn(torch.Size([1, 1536, 1, 1]))
x433 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x437, x433)
end = time.time()
print(end-start)
