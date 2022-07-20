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
        self.conv2d15 = Conv2d(128, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x57, x51, x65):
        x58=self.conv2d15(x57)
        x66=torch.cat([x51, x58, x65], 1)
        return x66

m = M().eval()
x57 = torch.randn(torch.Size([1, 128, 28, 28]))
x51 = torch.randn(torch.Size([1, 128, 28, 28]))
x65 = torch.randn(torch.Size([1, 32, 28, 28]))
start = time.time()
output = m(x57, x51, x65)
end = time.time()
print(end-start)
