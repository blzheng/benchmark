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
        self.conv2d102 = Conv2d(58, 1392, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid20 = Sigmoid()

    def forward(self, x314, x311):
        x315=self.conv2d102(x314)
        x316=self.sigmoid20(x315)
        x317=operator.mul(x316, x311)
        return x317

m = M().eval()
x314 = torch.randn(torch.Size([1, 58, 1, 1]))
x311 = torch.randn(torch.Size([1, 1392, 7, 7]))
start = time.time()
output = m(x314, x311)
end = time.time()
print(end-start)
