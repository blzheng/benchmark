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
        self.relu7 = ReLU()
        self.conv2d10 = Conv2d(18, 72, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()
        self.conv2d11 = Conv2d(72, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x29, x27):
        x30=self.relu7(x29)
        x31=self.conv2d10(x30)
        x32=self.sigmoid1(x31)
        x33=operator.mul(x32, x27)
        x34=self.conv2d11(x33)
        return x34

m = M().eval()
x29 = torch.randn(torch.Size([1, 18, 1, 1]))
x27 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x29, x27)
end = time.time()
print(end-start)
