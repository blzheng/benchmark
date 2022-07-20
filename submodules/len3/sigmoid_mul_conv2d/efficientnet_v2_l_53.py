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
        self.sigmoid53 = Sigmoid()
        self.conv2d302 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x969, x965):
        x970=self.sigmoid53(x969)
        x971=operator.mul(x970, x965)
        x972=self.conv2d302(x971)
        return x972

m = M().eval()
x969 = torch.randn(torch.Size([1, 2304, 1, 1]))
x965 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x969, x965)
end = time.time()
print(end-start)
