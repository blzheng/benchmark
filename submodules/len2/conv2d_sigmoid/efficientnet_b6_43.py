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
        self.conv2d216 = Conv2d(144, 3456, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid43 = Sigmoid()

    def forward(self, x677):
        x678=self.conv2d216(x677)
        x679=self.sigmoid43(x678)
        return x679

m = M().eval()
x677 = torch.randn(torch.Size([1, 144, 1, 1]))
start = time.time()
output = m(x677)
end = time.time()
print(end-start)
