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
        self.sigmoid31 = Sigmoid()
        self.conv2d158 = Conv2d(2688, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x489, x485):
        x490=self.sigmoid31(x489)
        x491=operator.mul(x490, x485)
        x492=self.conv2d158(x491)
        return x492

m = M().eval()
x489 = torch.randn(torch.Size([1, 2688, 1, 1]))
x485 = torch.randn(torch.Size([1, 2688, 7, 7]))
start = time.time()
output = m(x489, x485)
end = time.time()
print(end-start)
