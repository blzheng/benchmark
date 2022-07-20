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
        self.conv2d104 = Conv2d(256, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x330, x315):
        x331=operator.add(x330, x315)
        x332=self.conv2d104(x331)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 256, 7, 7]))
x315 = torch.randn(torch.Size([1, 256, 7, 7]))
start = time.time()
output = m(x330, x315)
end = time.time()
print(end-start)
