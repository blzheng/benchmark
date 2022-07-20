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
        self.conv2d88 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x255, x260):
        x261=operator.mul(x255, x260)
        x262=self.conv2d88(x261)
        return x262

m = M().eval()
x255 = torch.randn(torch.Size([1, 960, 14, 14]))
x260 = torch.randn(torch.Size([1, 960, 1, 1]))
start = time.time()
output = m(x255, x260)
end = time.time()
print(end-start)
