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
        self.conv2d28 = Conv2d(288, 288, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=288, bias=False)

    def forward(self, x86):
        x87=self.conv2d28(x86)
        return x87

m = M().eval()
x86 = torch.randn(torch.Size([1, 288, 56, 56]))
start = time.time()
output = m(x86)
end = time.time()
print(end-start)
