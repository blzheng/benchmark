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
        self.conv2d76 = Conv2d(480, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x223, x228):
        x229=operator.mul(x223, x228)
        x230=self.conv2d76(x229)
        return x230

m = M().eval()
x223 = torch.randn(torch.Size([1, 480, 28, 28]))
x228 = torch.randn(torch.Size([1, 480, 1, 1]))
start = time.time()
output = m(x223, x228)
end = time.time()
print(end-start)
