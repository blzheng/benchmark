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
        self.conv2d36 = Conv2d(174, 696, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d37 = Conv2d(696, 696, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x112, x109):
        x113=self.conv2d36(x112)
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        x116=self.conv2d37(x115)
        return x116

m = M().eval()
x112 = torch.randn(torch.Size([1, 174, 1, 1]))
x109 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x112, x109)
end = time.time()
print(end-start)
