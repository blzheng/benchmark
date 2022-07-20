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
        self.conv2d4 = Conv2d(128, 16, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x7, x9):
        x10=torch.cat([x7, x9], 1)
        x11=self.conv2d4(x10)
        return x11

m = M().eval()
x7 = torch.randn(torch.Size([1, 64, 54, 54]))
x9 = torch.randn(torch.Size([1, 64, 54, 54]))
start = time.time()
output = m(x7, x9)
end = time.time()
print(end-start)
