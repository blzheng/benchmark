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
        self.conv2d30 = Conv2d(120, 48, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x87):
        x88=self.conv2d30(x87)
        return x88

m = M().eval()
x87 = torch.randn(torch.Size([1, 120, 14, 14]))
start = time.time()
output = m(x87)
end = time.time()
print(end-start)