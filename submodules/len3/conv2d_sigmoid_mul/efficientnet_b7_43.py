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
        self.conv2d215 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid43 = Sigmoid()

    def forward(self, x676, x673):
        x677=self.conv2d215(x676)
        x678=self.sigmoid43(x677)
        x679=operator.mul(x678, x673)
        return x679

m = M().eval()
x676 = torch.randn(torch.Size([1, 96, 1, 1]))
x673 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x676, x673)
end = time.time()
print(end-start)
