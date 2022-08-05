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
        self.conv2d86 = Conv2d(2904, 726, kernel_size=(1, 1), stride=(1, 1))
        self.relu67 = ReLU()
        self.conv2d87 = Conv2d(726, 2904, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()
        self.conv2d88 = Conv2d(2904, 2904, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x272, x271):
        x273=self.conv2d86(x272)
        x274=self.relu67(x273)
        x275=self.conv2d87(x274)
        x276=self.sigmoid16(x275)
        x277=operator.mul(x276, x271)
        x278=self.conv2d88(x277)
        return x278

m = M().eval()
x272 = torch.randn(torch.Size([1, 2904, 1, 1]))
x271 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x272, x271)
end = time.time()
print(end-start)
