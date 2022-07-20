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
        self.conv2d58 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x166, x171):
        x172=operator.mul(x166, x171)
        x173=self.conv2d58(x172)
        return x173

m = M().eval()
x166 = torch.randn(torch.Size([1, 672, 14, 14]))
x171 = torch.randn(torch.Size([1, 672, 1, 1]))
start = time.time()
output = m(x166, x171)
end = time.time()
print(end-start)
