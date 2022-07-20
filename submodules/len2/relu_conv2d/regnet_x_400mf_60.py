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
        self.relu60 = ReLU(inplace=True)
        self.conv2d65 = Conv2d(400, 400, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x210):
        x211=self.relu60(x210)
        x212=self.conv2d65(x211)
        return x212

m = M().eval()
x210 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x210)
end = time.time()
print(end-start)
