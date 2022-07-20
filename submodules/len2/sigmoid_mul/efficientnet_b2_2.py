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
        self.sigmoid2 = Sigmoid()

    def forward(self, x37, x33):
        x38=self.sigmoid2(x37)
        x39=operator.mul(x38, x33)
        return x39

m = M().eval()
x37 = torch.randn(torch.Size([1, 96, 1, 1]))
x33 = torch.randn(torch.Size([1, 96, 56, 56]))
start = time.time()
output = m(x37, x33)
end = time.time()
print(end-start)
