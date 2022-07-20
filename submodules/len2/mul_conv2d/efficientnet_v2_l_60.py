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
        self.conv2d337 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1080, x1075):
        x1081=operator.mul(x1080, x1075)
        x1082=self.conv2d337(x1081)
        return x1082

m = M().eval()
x1080 = torch.randn(torch.Size([1, 3840, 1, 1]))
x1075 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1080, x1075)
end = time.time()
print(end-start)
