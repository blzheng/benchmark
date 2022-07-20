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
        self.sigmoid26 = Sigmoid()
        self.conv2d131 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x409, x405):
        x410=self.sigmoid26(x409)
        x411=operator.mul(x410, x405)
        x412=self.conv2d131(x411)
        return x412

m = M().eval()
x409 = torch.randn(torch.Size([1, 960, 1, 1]))
x405 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x409, x405)
end = time.time()
print(end-start)
