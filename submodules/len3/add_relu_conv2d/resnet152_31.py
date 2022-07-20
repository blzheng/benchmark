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
        self.relu94 = ReLU(inplace=True)
        self.conv2d100 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x328, x320):
        x329=operator.add(x328, x320)
        x330=self.relu94(x329)
        x331=self.conv2d100(x330)
        return x331

m = M().eval()
x328 = torch.randn(torch.Size([1, 1024, 14, 14]))
x320 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x328, x320)
end = time.time()
print(end-start)
