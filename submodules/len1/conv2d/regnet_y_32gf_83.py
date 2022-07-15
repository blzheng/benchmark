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
        self.conv2d83 = Conv2d(1392, 1392, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x261):
        x262=self.conv2d83(x261)
        return x262

m = M().eval()
x261 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x261)
end = time.time()
print(end-start)
