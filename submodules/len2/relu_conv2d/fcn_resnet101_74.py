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
        self.relu73 = ReLU(inplace=True)
        self.conv2d79 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x261):
        x262=self.relu73(x261)
        x263=self.conv2d79(x262)
        return x263

m = M().eval()
x261 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x261)
end = time.time()
print(end-start)
