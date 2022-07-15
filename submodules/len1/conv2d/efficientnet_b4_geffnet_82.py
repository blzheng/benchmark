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
        self.conv2d82 = Conv2d(28, 672, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x244):
        x245=self.conv2d82(x244)
        return x245

m = M().eval()
x244 = torch.randn(torch.Size([1, 28, 1, 1]))
start = time.time()
output = m(x244)
end = time.time()
print(end-start)
