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
        self.sigmoid1 = Sigmoid()

    def forward(self, x89, x85):
        x90=self.sigmoid1(x89)
        x91=operator.mul(x90, x85)
        return x91

m = M().eval()
x89 = torch.randn(torch.Size([1, 512, 1, 1]))
x85 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x89, x85)
end = time.time()
print(end-start)
