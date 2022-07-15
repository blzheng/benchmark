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
        self.conv2d12 = Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x34):
        x40=self.conv2d12(x34)
        return x40

m = M().eval()
x34 = torch.randn(torch.Size([1, 128, 28, 28]))
start = time.time()
output = m(x34)
end = time.time()
print(end-start)
