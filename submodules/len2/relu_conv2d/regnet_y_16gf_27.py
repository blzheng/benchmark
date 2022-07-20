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
        self.relu36 = ReLU(inplace=True)
        self.conv2d49 = Conv2d(1232, 1232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x152):
        x153=self.relu36(x152)
        x154=self.conv2d49(x153)
        return x154

m = M().eval()
x152 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
