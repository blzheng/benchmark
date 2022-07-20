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
        self.conv2d47 = Conv2d(32, 512, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid5 = Sigmoid()

    def forward(self, x152):
        x153=self.conv2d47(x152)
        x154=self.sigmoid5(x153)
        return x154

m = M().eval()
x152 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x152)
end = time.time()
print(end-start)
