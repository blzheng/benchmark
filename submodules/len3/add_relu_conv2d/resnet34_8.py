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
        self.relu17 = ReLU(inplace=True)
        self.conv2d21 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x69, x64):
        x70=operator.add(x69, x64)
        x71=self.relu17(x70)
        x72=self.conv2d21(x71)
        return x72

m = M().eval()
x69 = torch.randn(torch.Size([1, 256, 14, 14]))
x64 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x69, x64)
end = time.time()
print(end-start)
