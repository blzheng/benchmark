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
        self.conv2d46 = Conv2d(48, 768, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid2 = Sigmoid()
        self.conv2d47 = Conv2d(768, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x156, x153):
        x157=self.conv2d46(x156)
        x158=self.sigmoid2(x157)
        x159=operator.mul(x158, x153)
        x160=self.conv2d47(x159)
        return x160

m = M().eval()
x156 = torch.randn(torch.Size([1, 48, 1, 1]))
x153 = torch.randn(torch.Size([1, 768, 14, 14]))
start = time.time()
output = m(x156, x153)
end = time.time()
print(end-start)
