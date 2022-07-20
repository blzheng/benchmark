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
        self.conv2d230 = Conv2d(96, 2304, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid46 = Sigmoid()
        self.conv2d231 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x724, x721):
        x725=self.conv2d230(x724)
        x726=self.sigmoid46(x725)
        x727=operator.mul(x726, x721)
        x728=self.conv2d231(x727)
        return x728

m = M().eval()
x724 = torch.randn(torch.Size([1, 96, 1, 1]))
x721 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x724, x721)
end = time.time()
print(end-start)
