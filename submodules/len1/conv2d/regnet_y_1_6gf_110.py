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
        self.conv2d110 = Conv2d(336, 336, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=14, bias=False)

    def forward(self, x348):
        x349=self.conv2d110(x348)
        return x349

m = M().eval()
x348 = torch.randn(torch.Size([1, 336, 14, 14]))
start = time.time()
output = m(x348)
end = time.time()
print(end-start)
