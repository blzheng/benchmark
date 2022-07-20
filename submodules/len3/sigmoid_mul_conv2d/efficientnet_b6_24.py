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
        self.sigmoid24 = Sigmoid()
        self.conv2d122 = Conv2d(1200, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x378, x374):
        x379=self.sigmoid24(x378)
        x380=operator.mul(x379, x374)
        x381=self.conv2d122(x380)
        return x381

m = M().eval()
x378 = torch.randn(torch.Size([1, 1200, 1, 1]))
x374 = torch.randn(torch.Size([1, 1200, 14, 14]))
start = time.time()
output = m(x378, x374)
end = time.time()
print(end-start)
