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
        self.relu57 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x198):
        x199=self.relu57(x198)
        x200=self.conv2d61(x199)
        return x200

m = M().eval()
x198 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x198)
end = time.time()
print(end-start)
