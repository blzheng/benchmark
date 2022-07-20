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
        self.relu35 = ReLU()
        self.conv2d47 = Conv2d(348, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid8 = Sigmoid()

    def forward(self, x145, x143):
        x146=self.relu35(x145)
        x147=self.conv2d47(x146)
        x148=self.sigmoid8(x147)
        x149=operator.mul(x148, x143)
        return x149

m = M().eval()
x145 = torch.randn(torch.Size([1, 348, 1, 1]))
x143 = torch.randn(torch.Size([1, 1392, 14, 14]))
start = time.time()
output = m(x145, x143)
end = time.time()
print(end-start)