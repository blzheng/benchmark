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
        self.relu31 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x113):
        x114=self.relu31(x113)
        x115=self.conv2d35(x114)
        return x115

m = M().eval()
x113 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x113)
end = time.time()
print(end-start)
