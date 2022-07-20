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
        self.conv2d107 = Conv2d(1056, 176, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x313, x318):
        x319=operator.mul(x313, x318)
        x320=self.conv2d107(x319)
        return x320

m = M().eval()
x313 = torch.randn(torch.Size([1, 1056, 14, 14]))
x318 = torch.randn(torch.Size([1, 1056, 1, 1]))
start = time.time()
output = m(x313, x318)
end = time.time()
print(end-start)
