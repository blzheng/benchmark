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
        self.conv2d117 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x380, x375):
        x381=operator.mul(x380, x375)
        x382=self.conv2d117(x381)
        return x382

m = M().eval()
x380 = torch.randn(torch.Size([1, 1344, 1, 1]))
x375 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x380, x375)
end = time.time()
print(end-start)
