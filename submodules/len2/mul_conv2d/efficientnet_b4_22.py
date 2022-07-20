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
        self.conv2d113 = Conv2d(960, 272, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x350, x345):
        x351=operator.mul(x350, x345)
        x352=self.conv2d113(x351)
        return x352

m = M().eval()
x350 = torch.randn(torch.Size([1, 960, 1, 1]))
x345 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x350, x345)
end = time.time()
print(end-start)
