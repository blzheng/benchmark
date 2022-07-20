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
        self.relu184 = ReLU(inplace=True)
        self.conv2d184 = Conv2d(1664, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x651):
        x652=self.relu184(x651)
        x653=self.conv2d184(x652)
        return x653

m = M().eval()
x651 = torch.randn(torch.Size([1, 1664, 7, 7]))
start = time.time()
output = m(x651)
end = time.time()
print(end-start)
