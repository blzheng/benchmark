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
        self.conv2d111 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x331, x327):
        x332=x331.sigmoid()
        x333=operator.mul(x327, x332)
        x334=self.conv2d111(x333)
        return x334

m = M().eval()
x331 = torch.randn(torch.Size([1, 960, 1, 1]))
x327 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x331, x327)
end = time.time()
print(end-start)
