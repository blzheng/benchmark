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
        self.conv2d109 = Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x364):
        x365=self.conv2d109(x364)
        return x365

m = M().eval()
x364 = torch.randn(torch.Size([1, 1280, 28, 28]))
start = time.time()
output = m(x364)
end = time.time()
print(end-start)
