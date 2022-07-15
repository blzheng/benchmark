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
        self.conv2d62 = Conv2d(192, 192, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)

    def forward(self, x215):
        x216=self.conv2d62(x215)
        return x216

m = M().eval()
x215 = torch.randn(torch.Size([1, 192, 12, 12]))
start = time.time()
output = m(x215)
end = time.time()
print(end-start)
