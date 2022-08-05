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
        self.relu76 = ReLU(inplace=True)
        self.conv2d81 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x267):
        x268=self.relu76(x267)
        x269=self.conv2d81(x268)
        return x269

m = M().eval()
x267 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x267)
end = time.time()
print(end-start)
