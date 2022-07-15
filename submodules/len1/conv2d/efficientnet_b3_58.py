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
        self.conv2d58 = Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x177):
        x178=self.conv2d58(x177)
        return x178

m = M().eval()
x177 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x177)
end = time.time()
print(end-start)
