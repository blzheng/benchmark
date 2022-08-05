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
        self.conv2d65 = Conv2d(960, 256, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36), bias=False)

    def forward(self, x183):
        x193=self.conv2d65(x183)
        return x193

m = M().eval()
x183 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x183)
end = time.time()
print(end-start)
