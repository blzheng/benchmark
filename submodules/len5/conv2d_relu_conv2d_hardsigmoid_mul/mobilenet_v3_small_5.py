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
        self.conv2d33 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1))
        self.relu10 = ReLU()
        self.conv2d34 = Conv2d(40, 144, kernel_size=(1, 1), stride=(1, 1))
        self.hardsigmoid5 = Hardsigmoid()

    def forward(self, x96, x95):
        x97=self.conv2d33(x96)
        x98=self.relu10(x97)
        x99=self.conv2d34(x98)
        x100=self.hardsigmoid5(x99)
        x101=operator.mul(x100, x95)
        return x101

m = M().eval()
x96 = torch.randn(torch.Size([1, 144, 1, 1]))
x95 = torch.randn(torch.Size([1, 144, 14, 14]))
start = time.time()
output = m(x96, x95)
end = time.time()
print(end-start)
