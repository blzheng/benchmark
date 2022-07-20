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
        self.sigmoid4 = Sigmoid()
        self.conv2d43 = Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x137, x133):
        x138=self.sigmoid4(x137)
        x139=operator.mul(x138, x133)
        x140=self.conv2d43(x139)
        return x140

m = M().eval()
x137 = torch.randn(torch.Size([1, 512, 1, 1]))
x133 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x137, x133)
end = time.time()
print(end-start)
