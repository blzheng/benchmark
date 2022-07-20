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
        self.sigmoid45 = Sigmoid()
        self.conv2d226 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x709, x705):
        x710=self.sigmoid45(x709)
        x711=operator.mul(x710, x705)
        x712=self.conv2d226(x711)
        return x712

m = M().eval()
x709 = torch.randn(torch.Size([1, 2304, 1, 1]))
x705 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x709, x705)
end = time.time()
print(end-start)
