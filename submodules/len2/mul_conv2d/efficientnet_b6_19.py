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
        self.conv2d97 = Conv2d(864, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x301, x296):
        x302=operator.mul(x301, x296)
        x303=self.conv2d97(x302)
        return x303

m = M().eval()
x301 = torch.randn(torch.Size([1, 864, 1, 1]))
x296 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x301, x296)
end = time.time()
print(end-start)
