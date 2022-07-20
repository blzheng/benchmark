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
        self.conv2d117 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()
        self.conv2d118 = Conv2d(1392, 232, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x362, x359):
        x363=self.conv2d117(x362)
        x364=self.sigmoid23(x363)
        x365=operator.mul(x364, x359)
        x366=self.conv2d118(x365)
        return x366

m = M().eval()
x362 = torch.randn(torch.Size([1, 58, 1, 1]))
x359 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x362, x359)
end = time.time()
print(end-start)
