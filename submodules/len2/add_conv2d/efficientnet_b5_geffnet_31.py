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
        self.conv2d193 = Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x574, x560):
        x575=operator.add(x574, x560)
        x576=self.conv2d193(x575)
        return x576

m = M().eval()
x574 = torch.randn(torch.Size([1, 512, 7, 7]))
x560 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x574, x560)
end = time.time()
print(end-start)
