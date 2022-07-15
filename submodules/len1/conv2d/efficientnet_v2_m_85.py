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
        self.conv2d85 = Conv2d(1056, 1056, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1056, bias=False)

    def forward(self, x275):
        x276=self.conv2d85(x275)
        return x276

m = M().eval()
x275 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x275)
end = time.time()
print(end-start)
