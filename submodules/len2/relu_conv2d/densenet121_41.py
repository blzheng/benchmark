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
        self.relu42 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x152):
        x153=self.relu42(x152)
        x154=self.conv2d42(x153)
        return x154

m = M().eval()
x152 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
