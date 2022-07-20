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
        self.relu55 = ReLU(inplace=True)
        self.conv2d61 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x198, x190):
        x199=operator.add(x198, x190)
        x200=self.relu55(x199)
        x201=self.conv2d61(x200)
        return x201

m = M().eval()
x198 = torch.randn(torch.Size([1, 1024, 14, 14]))
x190 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x198, x190)
end = time.time()
print(end-start)
