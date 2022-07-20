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
        self.sigmoid7 = Sigmoid()
        self.conv2d43 = Conv2d(320, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x131, x127):
        x132=self.sigmoid7(x131)
        x133=operator.mul(x132, x127)
        x134=self.conv2d43(x133)
        return x134

m = M().eval()
x131 = torch.randn(torch.Size([1, 320, 1, 1]))
x127 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x131, x127)
end = time.time()
print(end-start)
