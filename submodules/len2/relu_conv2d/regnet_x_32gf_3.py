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
        self.relu3 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x14):
        x15=self.relu3(x14)
        x16=self.conv2d5(x15)
        return x16

m = M().eval()
x14 = torch.randn(torch.Size([1, 336, 56, 56]))
start = time.time()
output = m(x14)
end = time.time()
print(end-start)
