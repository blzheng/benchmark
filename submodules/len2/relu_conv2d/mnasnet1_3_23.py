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
        self.relu23 = ReLU(inplace=True)
        self.conv2d35 = Conv2d(768, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x99):
        x100=self.relu23(x99)
        x101=self.conv2d35(x100)
        return x101

m = M().eval()
x99 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)
