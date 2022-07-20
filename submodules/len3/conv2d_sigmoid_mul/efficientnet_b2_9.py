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
        self.conv2d47 = Conv2d(22, 528, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid9 = Sigmoid()

    def forward(self, x142, x139):
        x143=self.conv2d47(x142)
        x144=self.sigmoid9(x143)
        x145=operator.mul(x144, x139)
        return x145

m = M().eval()
x142 = torch.randn(torch.Size([1, 22, 1, 1]))
x139 = torch.randn(torch.Size([1, 528, 14, 14]))
start = time.time()
output = m(x142, x139)
end = time.time()
print(end-start)
