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
        self.conv2d102 = Conv2d(64, 1536, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid16 = Sigmoid()
        self.conv2d103 = Conv2d(1536, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x324, x321):
        x325=self.conv2d102(x324)
        x326=self.sigmoid16(x325)
        x327=operator.mul(x326, x321)
        x328=self.conv2d103(x327)
        return x328

m = M().eval()
x324 = torch.randn(torch.Size([1, 64, 1, 1]))
x321 = torch.randn(torch.Size([1, 1536, 7, 7]))
start = time.time()
output = m(x324, x321)
end = time.time()
print(end-start)
