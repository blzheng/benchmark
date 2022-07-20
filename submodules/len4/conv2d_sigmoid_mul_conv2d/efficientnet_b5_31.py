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
        self.conv2d156 = Conv2d(76, 1824, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid31 = Sigmoid()
        self.conv2d157 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x487, x484):
        x488=self.conv2d156(x487)
        x489=self.sigmoid31(x488)
        x490=operator.mul(x489, x484)
        x491=self.conv2d157(x490)
        return x491

m = M().eval()
x487 = torch.randn(torch.Size([1, 76, 1, 1]))
x484 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x487, x484)
end = time.time()
print(end-start)
