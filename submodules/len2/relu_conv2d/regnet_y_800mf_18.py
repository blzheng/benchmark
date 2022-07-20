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
        self.relu24 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x104):
        x105=self.relu24(x104)
        x106=self.conv2d34(x105)
        return x106

m = M().eval()
x104 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x104)
end = time.time()
print(end-start)
