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
        self.conv2d106 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid14 = Sigmoid()

    def forward(self, x346, x343):
        x347=self.conv2d106(x346)
        x348=self.sigmoid14(x347)
        x349=operator.mul(x348, x343)
        return x349

m = M().eval()
x346 = torch.randn(torch.Size([1, 56, 1, 1]))
x343 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x346, x343)
end = time.time()
print(end-start)
