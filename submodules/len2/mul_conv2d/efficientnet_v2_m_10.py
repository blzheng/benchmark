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
        self.conv2d78 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x251, x246):
        x252=operator.mul(x251, x246)
        x253=self.conv2d78(x252)
        return x253

m = M().eval()
x251 = torch.randn(torch.Size([1, 1056, 1, 1]))
x246 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x251, x246)
end = time.time()
print(end-start)
