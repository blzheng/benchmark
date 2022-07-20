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
        self.conv2d123 = Conv2d(1392, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x363, x359):
        x364=x363.sigmoid()
        x365=operator.mul(x359, x364)
        x366=self.conv2d123(x365)
        return x366

m = M().eval()
x363 = torch.randn(torch.Size([1, 1392, 1, 1]))
x359 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x363, x359)
end = time.time()
print(end-start)
