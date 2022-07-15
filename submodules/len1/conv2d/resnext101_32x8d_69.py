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
        self.conv2d69 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x226):
        x227=self.conv2d69(x226)
        return x227

m = M().eval()
x226 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x226)
end = time.time()
print(end-start)
