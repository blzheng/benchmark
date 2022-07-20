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
        self.conv2d76 = Conv2d(18, 432, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid15 = Sigmoid()

    def forward(self, x237, x234):
        x238=self.conv2d76(x237)
        x239=self.sigmoid15(x238)
        x240=operator.mul(x239, x234)
        return x240

m = M().eval()
x237 = torch.randn(torch.Size([1, 18, 1, 1]))
x234 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x237, x234)
end = time.time()
print(end-start)
