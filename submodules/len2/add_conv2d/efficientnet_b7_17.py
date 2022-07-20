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
        self.conv2d107 = Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x334, x319):
        x335=operator.add(x334, x319)
        x336=self.conv2d107(x335)
        return x336

m = M().eval()
x334 = torch.randn(torch.Size([1, 160, 14, 14]))
x319 = torch.randn(torch.Size([1, 160, 14, 14]))
start = time.time()
output = m(x334, x319)
end = time.time()
print(end-start)
