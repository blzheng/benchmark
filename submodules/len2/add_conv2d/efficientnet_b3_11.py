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
        self.conv2d84 = Conv2d(136, 816, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x258, x243):
        x259=operator.add(x258, x243)
        x260=self.conv2d84(x259)
        return x260

m = M().eval()
x258 = torch.randn(torch.Size([1, 136, 14, 14]))
x243 = torch.randn(torch.Size([1, 136, 14, 14]))
start = time.time()
output = m(x258, x243)
end = time.time()
print(end-start)
