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
        self.conv2d94 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x300, x285):
        x301=operator.add(x300, x285)
        x302=self.conv2d94(x301)
        return x302

m = M().eval()
x300 = torch.randn(torch.Size([1, 160, 14, 14]))
x285 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x300, x285)
end = time.time()
print(end-start)
