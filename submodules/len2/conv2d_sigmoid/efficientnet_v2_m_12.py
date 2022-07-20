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
        self.conv2d87 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x281):
        x282=self.conv2d87(x281)
        x283=self.sigmoid12(x282)
        return x283

m = M().eval()
x281 = torch.randn(torch.Size([1, 44, 1, 1]))
start = time.time()
output = m(x281)
end = time.time()
print(end-start)
