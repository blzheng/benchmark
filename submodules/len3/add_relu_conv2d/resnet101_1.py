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
        self.relu4 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x24, x16):
        x25=operator.add(x24, x16)
        x26=self.relu4(x25)
        x27=self.conv2d8(x26)
        return x27

m = M().eval()
x24 = torch.randn(torch.Size([1, 256, 56, 56]))
x16 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x24, x16)
end = time.time()
print(end-start)
