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
        self.conv2d61 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x179, x165):
        x180=operator.add(x179, x165)
        x181=self.conv2d61(x180)
        return x181

m = M().eval()
x179 = torch.randn(torch.Size([1, 160, 14, 14]))
x165 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x179, x165)
end = time.time()
print(end-start)
