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
        self.conv2d112 = Conv2d(1344, 224, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x364, x359):
        x365=operator.mul(x364, x359)
        x366=self.conv2d112(x365)
        return x366

m = M().eval()
x364 = torch.randn(torch.Size([1, 1344, 1, 1]))
x359 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x364, x359)
end = time.time()
print(end-start)