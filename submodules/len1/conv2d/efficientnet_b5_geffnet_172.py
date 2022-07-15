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
        self.conv2d172 = Conv2d(1824, 304, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x513):
        x514=self.conv2d172(x513)
        return x514

m = M().eval()
x513 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x513)
end = time.time()
print(end-start)
