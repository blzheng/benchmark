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
        self.conv2d83 = Conv2d(672, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x254, x249):
        x255=operator.mul(x254, x249)
        x256=self.conv2d83(x255)
        return x256

m = M().eval()
x254 = torch.randn(torch.Size([1, 672, 1, 1]))
x249 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x254, x249)
end = time.time()
print(end-start)
