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
        self.conv2d103 = Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x319):
        x320=self.conv2d103(x319)
        return x320

m = M().eval()
x319 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x319)
end = time.time()
print(end-start)
