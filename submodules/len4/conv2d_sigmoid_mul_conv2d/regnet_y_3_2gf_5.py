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
        self.conv2d31 = Conv2d(54, 216, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()
        self.conv2d32 = Conv2d(216, 216, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x96, x93):
        x97=self.conv2d31(x96)
        x98=self.sigmoid5(x97)
        x99=operator.mul(x98, x93)
        x100=self.conv2d32(x99)
        return x100

m = M().eval()
x96 = torch.randn(torch.Size([1, 54, 1, 1]))
x93 = torch.randn(torch.Size([1, 216, 28, 28]))
start = time.time()
output = m(x96, x93)
end = time.time()
print(end-start)
