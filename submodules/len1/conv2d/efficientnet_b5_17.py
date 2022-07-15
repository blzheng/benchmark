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
        self.conv2d17 = Conv2d(144, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x52):
        x53=self.conv2d17(x52)
        return x53

m = M().eval()
x52 = torch.randn(torch.Size([1, 144, 56, 56]))
start = time.time()
output = m(x52)
end = time.time()
print(end-start)
