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
        self.conv2d130 = Conv2d(1200, 50, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x407):
        x408=self.conv2d130(x407)
        return x408

m = M().eval()
x407 = torch.randn(torch.Size([1, 1200, 1, 1]))
start = time.time()
output = m(x407)
end = time.time()
print(end-start)
