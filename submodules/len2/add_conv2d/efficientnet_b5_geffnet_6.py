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
        self.conv2d48 = Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x143, x129):
        x144=operator.add(x143, x129)
        x145=self.conv2d48(x144)
        return x145

m = M().eval()
x143 = torch.randn(torch.Size([1, 64, 28, 28]))
x129 = torch.randn(torch.Size([1, 64, 28, 28]))
start = time.time()
output = m(x143, x129)
end = time.time()
print(end-start)
