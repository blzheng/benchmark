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
        self.conv2d94 = Conv2d(208, 1248, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x288, x273):
        x289=operator.add(x288, x273)
        x290=self.conv2d94(x289)
        return x290

m = M().eval()
x288 = torch.randn(torch.Size([1, 208, 7, 7]))
x273 = torch.randn(torch.Size([1, 208, 7, 7]))
start = time.time()
output = m(x288, x273)
end = time.time()
print(end-start)
