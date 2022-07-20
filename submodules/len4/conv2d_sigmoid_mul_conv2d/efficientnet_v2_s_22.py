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
        self.conv2d132 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid22 = Sigmoid()
        self.conv2d133 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x420, x417):
        x421=self.conv2d132(x420)
        x422=self.sigmoid22(x421)
        x423=operator.mul(x422, x417)
        x424=self.conv2d133(x423)
        return x424

m = M().eval()
x420 = torch.randn(torch.Size([1, 64, 1, 1]))
x417 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x420, x417)
end = time.time()
print(end-start)
