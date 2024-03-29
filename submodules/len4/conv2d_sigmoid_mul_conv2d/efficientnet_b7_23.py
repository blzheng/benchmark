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
        self.conv2d115 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()
        self.conv2d116 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x360, x357):
        x361=self.conv2d115(x360)
        x362=self.sigmoid23(x361)
        x363=operator.mul(x362, x357)
        x364=self.conv2d116(x363)
        return x364

m = M().eval()
x360 = torch.randn(torch.Size([1, 40, 1, 1]))
x357 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x360, x357)
end = time.time()
print(end-start)
