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
        self.conv2d73 = Conv2d(192, 768, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x243):
        x244=self.conv2d73(x243)
        return x244

m = M().eval()
x243 = torch.randn(torch.Size([1, 192, 14, 14]))
start = time.time()
output = m(x243)
end = time.time()
print(end-start)
