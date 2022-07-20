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
        self.conv2d38 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x111, x107):
        x112=x111.sigmoid()
        x113=operator.mul(x107, x112)
        x114=self.conv2d38(x113)
        return x114

m = M().eval()
x111 = torch.randn(torch.Size([1, 240, 1, 1]))
x107 = torch.randn(torch.Size([1, 240, 28, 28]))
start = time.time()
output = m(x111, x107)
end = time.time()
print(end-start)
