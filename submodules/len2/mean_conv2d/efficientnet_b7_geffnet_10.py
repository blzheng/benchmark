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
        self.conv2d49 = Conv2d(288, 12, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x149):
        x150=x149.mean((2, 3),keepdim=True)
        x151=self.conv2d49(x150)
        return x151

m = M().eval()
x149 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x149)
end = time.time()
print(end-start)
