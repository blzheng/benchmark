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
        self.conv2d4 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x10):
        x11=self.relu2(x10)
        x12=self.conv2d4(x11)
        return x12

m = M().eval()
x10 = torch.randn(torch.Size([1, 336, 56, 56]))
start = time.time()
output = m(x10)
end = time.time()
print(end-start)
