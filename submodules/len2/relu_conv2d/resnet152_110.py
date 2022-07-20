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
        self.relu109 = ReLU(inplace=True)
        self.conv2d115 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x379):
        x380=self.relu109(x379)
        x381=self.conv2d115(x380)
        return x381

m = M().eval()
x379 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x379)
end = time.time()
print(end-start)