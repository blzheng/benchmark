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
        self.conv2d65 = Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x213):
        x214=self.conv2d65(x213)
        return x214

m = M().eval()
x213 = torch.randn(torch.Size([1, 512, 14, 14]))
start = time.time()
output = m(x213)
end = time.time()
print(end-start)