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
        self.conv2d162 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid28 = Sigmoid()

    def forward(self, x516):
        x517=self.conv2d162(x516)
        x518=self.sigmoid28(x517)
        return x518

m = M().eval()
x516 = torch.randn(torch.Size([1, 64, 1, 1]))
start = time.time()
output = m(x516)
end = time.time()
print(end-start)
