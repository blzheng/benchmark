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
        self.conv2d73 = Conv2d(816, 136, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x214, x210):
        x215=x214.sigmoid()
        x216=operator.mul(x210, x215)
        x217=self.conv2d73(x216)
        return x217

m = M().eval()
x214 = torch.randn(torch.Size([1, 816, 1, 1]))
x210 = torch.randn(torch.Size([1, 816, 14, 14]))
start = time.time()
output = m(x214, x210)
end = time.time()
print(end-start)
