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
        self.relu2 = ReLU(inplace=True)
        self.conv2d5 = Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x16):
        x17=self.relu2(x16)
        x18=self.conv2d5(x17)
        return x18

m = M().eval()
x16 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x16)
end = time.time()
print(end-start)
