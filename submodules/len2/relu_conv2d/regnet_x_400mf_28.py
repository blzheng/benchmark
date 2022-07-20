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
        self.relu28 = ReLU(inplace=True)
        self.conv2d32 = Conv2d(160, 160, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=10, bias=False)

    def forward(self, x101):
        x102=self.relu28(x101)
        x103=self.conv2d32(x102)
        return x103

m = M().eval()
x101 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x101)
end = time.time()
print(end-start)
