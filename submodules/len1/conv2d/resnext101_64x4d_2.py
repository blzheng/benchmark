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
        self.conv2d2 = Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)

    def forward(self, x7):
        x8=self.conv2d2(x7)
        return x8

m = M().eval()
x7 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x7)
end = time.time()
print(end-start)
