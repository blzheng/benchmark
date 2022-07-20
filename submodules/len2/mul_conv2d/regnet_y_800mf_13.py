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
        self.conv2d74 = Conv2d(784, 784, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x230, x225):
        x231=operator.mul(x230, x225)
        x232=self.conv2d74(x231)
        return x232

m = M().eval()
x230 = torch.randn(torch.Size([1, 784, 1, 1]))
x225 = torch.randn(torch.Size([1, 784, 7, 7]))
start = time.time()
output = m(x230, x225)
end = time.time()
print(end-start)
