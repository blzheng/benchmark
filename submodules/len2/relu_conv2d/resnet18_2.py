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
        self.relu3 = ReLU(inplace=True)
        self.conv2d4 = Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x13):
        x14=self.relu3(x13)
        x15=self.conv2d4(x14)
        return x15

m = M().eval()
x13 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x13)
end = time.time()
print(end-start)
