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
        self.relu127 = ReLU(inplace=True)
        self.conv2d133 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x438, x430):
        x439=operator.add(x438, x430)
        x440=self.relu127(x439)
        x441=self.conv2d133(x440)
        return x441

m = M().eval()
x438 = torch.randn(torch.Size([1, 1024, 14, 14]))
x430 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x438, x430)
end = time.time()
print(end-start)
