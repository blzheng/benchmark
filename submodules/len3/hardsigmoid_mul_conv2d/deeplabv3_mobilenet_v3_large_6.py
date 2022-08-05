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
        self.hardsigmoid6 = Hardsigmoid()
        self.conv2d55 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x160, x156):
        x161=self.hardsigmoid6(x160)
        x162=operator.mul(x161, x156)
        x163=self.conv2d55(x162)
        return x163

m = M().eval()
x160 = torch.randn(torch.Size([1, 960, 1, 1]))
x156 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x160, x156)
end = time.time()
print(end-start)
