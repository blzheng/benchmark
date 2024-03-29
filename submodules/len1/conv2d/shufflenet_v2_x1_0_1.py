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
        self.conv2d1 = Conv2d(24, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=24, bias=False)

    def forward(self, x4):
        x5=self.conv2d1(x4)
        return x5

m = M().eval()
x4 = torch.randn(torch.Size([1, 24, 56, 56]))
start = time.time()
output = m(x4)
end = time.time()
print(end-start)
