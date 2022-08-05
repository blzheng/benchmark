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
        self.relu28 = ReLU(inplace=True)
        self.conv2d34 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x110, x102):
        x111=operator.add(x110, x102)
        x112=self.relu28(x111)
        x113=self.conv2d34(x112)
        return x113

m = M().eval()
x110 = torch.randn(torch.Size([1, 1024, 28, 28]))
x102 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x110, x102)
end = time.time()
print(end-start)
