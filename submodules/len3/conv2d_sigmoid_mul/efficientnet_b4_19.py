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
        self.conv2d97 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid19 = Sigmoid()

    def forward(self, x300, x297):
        x301=self.conv2d97(x300)
        x302=self.sigmoid19(x301)
        x303=operator.mul(x302, x297)
        return x303

m = M().eval()
x300 = torch.randn(torch.Size([1, 40, 1, 1]))
x297 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x300, x297)
end = time.time()
print(end-start)
