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
        self.conv2d181 = Conv2d(86, 2064, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid36 = Sigmoid()
        self.conv2d182 = Conv2d(2064, 344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x567, x564):
        x568=self.conv2d181(x567)
        x569=self.sigmoid36(x568)
        x570=operator.mul(x569, x564)
        x571=self.conv2d182(x570)
        return x571

m = M().eval()
x567 = torch.randn(torch.Size([1, 86, 1, 1]))
x564 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x567, x564)
end = time.time()
print(end-start)
