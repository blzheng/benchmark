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
        self.conv2d317 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x1016, x1011):
        x1017=operator.mul(x1016, x1011)
        x1018=self.conv2d317(x1017)
        return x1018

m = M().eval()
x1016 = torch.randn(torch.Size([1, 3840, 1, 1]))
x1011 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x1016, x1011)
end = time.time()
print(end-start)
