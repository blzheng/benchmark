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
        self.sigmoid24 = Sigmoid()
        self.conv2d143 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x453, x449):
        x454=self.sigmoid24(x453)
        x455=operator.mul(x454, x449)
        x456=self.conv2d143(x455)
        return x456

m = M().eval()
x453 = torch.randn(torch.Size([1, 1536, 1, 1]))
x449 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x453, x449)
end = time.time()
print(end-start)
