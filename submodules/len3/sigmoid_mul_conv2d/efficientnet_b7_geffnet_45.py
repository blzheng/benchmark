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
        self.conv2d226 = Conv2d(2304, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x674, x670):
        x675=x674.sigmoid()
        x676=operator.mul(x670, x675)
        x677=self.conv2d226(x676)
        return x677

m = M().eval()
x674 = torch.randn(torch.Size([1, 2304, 1, 1]))
x670 = torch.randn(torch.Size([1, 2304, 7, 7]))
start = time.time()
output = m(x674, x670)
end = time.time()
print(end-start)
