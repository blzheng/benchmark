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
        self.conv2d96 = Conv2d(36, 864, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()

    def forward(self, x299, x296):
        x300=self.conv2d96(x299)
        x301=self.sigmoid19(x300)
        x302=operator.mul(x301, x296)
        return x302

m = M().eval()
x299 = torch.randn(torch.Size([1, 36, 1, 1]))
x296 = torch.randn(torch.Size([1, 864, 14, 14]))
start = time.time()
output = m(x299, x296)
end = time.time()
print(end-start)
