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
        self.conv2d63 = Conv2d(72, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x188, x174):
        x189=operator.add(x188, x174)
        x190=self.conv2d63(x189)
        return x190

m = M().eval()
x188 = torch.randn(torch.Size([1, 72, 28, 28]))
x174 = torch.randn(torch.Size([1, 72, 28, 28]))
start = time.time()
output = m(x188, x174)
end = time.time()
print(end-start)
