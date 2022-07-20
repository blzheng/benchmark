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
        self.relu16 = ReLU(inplace=True)
        self.conv2d17 = Conv2d(48, 192, kernel_size=(1, 1), stride=(1, 1))
        self.relu17 = ReLU(inplace=True)

    def forward(self, x40):
        x41=self.relu16(x40)
        x42=self.conv2d17(x41)
        x43=self.relu17(x42)
        return x43

m = M().eval()
x40 = torch.randn(torch.Size([1, 48, 27, 27]))
start = time.time()
output = m(x40)
end = time.time()
print(end-start)
