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
        self.conv2d67 = Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x201, x187):
        x202=operator.add(x201, x187)
        x203=self.conv2d67(x202)
        return x203

m = M().eval()
x201 = torch.randn(torch.Size([1, 80, 28, 28]))
x187 = torch.randn(torch.Size([1, 80, 28, 28]))
start = time.time()
output = m(x201, x187)
end = time.time()
print(end-start)
