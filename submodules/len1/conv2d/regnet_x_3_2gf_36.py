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
        self.conv2d36 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x115):
        x116=self.conv2d36(x115)
        return x116

m = M().eval()
x115 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x115)
end = time.time()
print(end-start)
