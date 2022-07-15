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
        self.conv2d71 = Conv2d(192, 320, kernel_size=(3, 3), stride=(2, 2), bias=False)

    def forward(self, x244):
        x245=self.conv2d71(x244)
        return x245

m = M().eval()
x244 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x244)
end = time.time()
print(end-start)
