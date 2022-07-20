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
        self.sigmoid2 = Sigmoid()
        self.conv2d17 = Conv2d(448, 448, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x49, x45):
        x50=self.sigmoid2(x49)
        x51=operator.mul(x50, x45)
        x52=self.conv2d17(x51)
        return x52

m = M().eval()
x49 = torch.randn(torch.Size([1, 448, 1, 1]))
x45 = torch.randn(torch.Size([1, 448, 28, 28]))
start = time.time()
output = m(x49, x45)
end = time.time()
print(end-start)
