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
        self.conv2d151 = Conv2d(56, 1344, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid23 = Sigmoid()

    def forward(self, x490):
        x491=self.conv2d151(x490)
        x492=self.sigmoid23(x491)
        return x492

m = M().eval()
x490 = torch.randn(torch.Size([1, 56, 1, 1]))
start = time.time()
output = m(x490)
end = time.time()
print(end-start)
