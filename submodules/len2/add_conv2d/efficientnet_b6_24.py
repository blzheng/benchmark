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
        self.conv2d148 = Conv2d(200, 1200, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x463, x448):
        x464=operator.add(x463, x448)
        x465=self.conv2d148(x464)
        return x465

m = M().eval()
x463 = torch.randn(torch.Size([1, 200, 14, 14]))
x448 = torch.randn(torch.Size([1, 200, 14, 14]))
start = time.time()
output = m(x463, x448)
end = time.time()
print(end-start)
