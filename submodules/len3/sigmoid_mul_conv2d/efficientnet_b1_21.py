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
        self.sigmoid21 = Sigmoid()
        self.conv2d108 = Conv2d(1152, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x331, x327):
        x332=self.sigmoid21(x331)
        x333=operator.mul(x332, x327)
        x334=self.conv2d108(x333)
        return x334

m = M().eval()
x331 = torch.randn(torch.Size([1, 1152, 1, 1]))
x327 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x331, x327)
end = time.time()
print(end-start)
