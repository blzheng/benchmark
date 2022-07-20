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
        self.sigmoid5 = Sigmoid()

    def forward(self, x77, x73):
        x78=self.sigmoid5(x77)
        x79=operator.mul(x78, x73)
        return x79

m = M().eval()
x77 = torch.randn(torch.Size([1, 288, 1, 1]))
x73 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x77, x73)
end = time.time()
print(end-start)