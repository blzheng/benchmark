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
        self.conv2d31 = Conv2d(10, 240, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid6 = Sigmoid()
        self.conv2d32 = Conv2d(240, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x95, x92):
        x96=self.conv2d31(x95)
        x97=self.sigmoid6(x96)
        x98=operator.mul(x97, x92)
        x99=self.conv2d32(x98)
        return x99

m = M().eval()
x95 = torch.randn(torch.Size([1, 10, 1, 1]))
x92 = torch.randn(torch.Size([1, 240, 56, 56]))
start = time.time()
output = m(x95, x92)
end = time.time()
print(end-start)
