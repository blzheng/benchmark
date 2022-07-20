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
        self.conv2d96 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x314, x311):
        x315=self.conv2d96(x314)
        x316=self.sigmoid12(x315)
        x317=operator.mul(x316, x311)
        return x317

m = M().eval()
x314 = torch.randn(torch.Size([1, 56, 1, 1]))
x311 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x314, x311)
end = time.time()
print(end-start)
