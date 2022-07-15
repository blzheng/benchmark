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
        self.conv2d33 = Conv2d(696, 696, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x103):
        x104=self.conv2d33(x103)
        return x104

m = M().eval()
x103 = torch.randn(torch.Size([1, 696, 28, 28]))
start = time.time()
output = m(x103)
end = time.time()
print(end-start)
