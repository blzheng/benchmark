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
        self.conv2d225 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid45 = Sigmoid()

    def forward(self, x708, x705):
        x709=self.conv2d225(x708)
        x710=self.sigmoid45(x709)
        x711=operator.mul(x710, x705)
        return x711

m = M().eval()
x708 = torch.randn(torch.Size([1, 96, 1, 1]))
x705 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x708, x705)
end = time.time()
print(end-start)
