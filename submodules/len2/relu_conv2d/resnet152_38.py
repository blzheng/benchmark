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
        self.relu37 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x139):
        x140=self.relu37(x139)
        x141=self.conv2d43(x140)
        return x141

m = M().eval()
x139 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x139)
end = time.time()
print(end-start)
