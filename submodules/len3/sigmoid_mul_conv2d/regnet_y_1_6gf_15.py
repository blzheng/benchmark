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
        self.sigmoid15 = Sigmoid()
        self.conv2d83 = Conv2d(336, 336, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x259, x255):
        x260=self.sigmoid15(x259)
        x261=operator.mul(x260, x255)
        x262=self.conv2d83(x261)
        return x262

m = M().eval()
x259 = torch.randn(torch.Size([1, 336, 1, 1]))
x255 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x259, x255)
end = time.time()
print(end-start)
