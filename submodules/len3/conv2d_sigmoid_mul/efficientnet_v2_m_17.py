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
        self.conv2d112 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid17 = Sigmoid()

    def forward(self, x361, x358):
        x362=self.conv2d112(x361)
        x363=self.sigmoid17(x362)
        x364=operator.mul(x363, x358)
        return x364

m = M().eval()
x361 = torch.randn(torch.Size([1, 44, 1, 1]))
x358 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x361, x358)
end = time.time()
print(end-start)
