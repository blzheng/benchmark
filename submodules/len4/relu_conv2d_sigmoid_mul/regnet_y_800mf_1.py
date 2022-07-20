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
        self.conv2d11 = Conv2d(16, 144, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid1 = Sigmoid()

    def forward(self, x31, x29):
        x32=self.relu7(x31)
        x33=self.conv2d11(x32)
        x34=self.sigmoid1(x33)
        x35=operator.mul(x34, x29)
        return x35

m = M().eval()
x31 = torch.randn(torch.Size([1, 16, 1, 1]))
x29 = torch.randn(torch.Size([1, 144, 28, 28]))
start = time.time()
output = m(x31, x29)
end = time.time()
print(end-start)
