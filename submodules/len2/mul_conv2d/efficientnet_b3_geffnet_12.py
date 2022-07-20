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
        self.conv2d63 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x181, x186):
        x187=operator.mul(x181, x186)
        x188=self.conv2d63(x187)
        return x188

m = M().eval()
x181 = torch.randn(torch.Size([1, 576, 14, 14]))
x186 = torch.randn(torch.Size([1, 576, 1, 1]))
start = time.time()
output = m(x181, x186)
end = time.time()
print(end-start)
