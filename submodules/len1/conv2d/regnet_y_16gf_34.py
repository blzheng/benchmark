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
        self.conv2d34 = Conv2d(448, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x103):
        x106=self.conv2d34(x103)
        return x106

m = M().eval()
x103 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
