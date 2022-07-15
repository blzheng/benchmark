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
        self.conv2d152 = Conv2d(1152, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x540):
        x541=self.conv2d152(x540)
        return x541

m = M().eval()
x540 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x540)
end = time.time()
print(end-start)
