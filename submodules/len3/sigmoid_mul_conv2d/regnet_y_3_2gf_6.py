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
        self.sigmoid6 = Sigmoid()
        self.conv2d37 = Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x113, x109):
        x114=self.sigmoid6(x113)
        x115=operator.mul(x114, x109)
        x116=self.conv2d37(x115)
        return x116

m = M().eval()
x113 = torch.randn(torch.Size([1, 216, 1, 1]))
x109 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x113, x109)
end = time.time()
print(end-start)
