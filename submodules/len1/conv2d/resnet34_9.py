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
        self.conv2d9 = Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)

    def forward(self, x25):
        x31=self.conv2d9(x25)
        return x31

m = M().eval()
x25 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x25)
end = time.time()
print(end-start)
