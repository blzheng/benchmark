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
        self.conv2d75 = Conv2d(20, 480, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()

    def forward(self, x234, x231):
        x235=self.conv2d75(x234)
        x236=self.sigmoid15(x235)
        x237=operator.mul(x236, x231)
        return x237

m = M().eval()
x234 = torch.randn(torch.Size([1, 20, 1, 1]))
x231 = torch.randn(torch.Size([1, 480, 28, 28]))
start = time.time()
output = m(x234, x231)
end = time.time()
print(end-start)
