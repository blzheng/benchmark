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
        self.conv2d39 = Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x126, x111):
        x127=operator.add(x126, x111)
        x128=self.conv2d39(x127)
        return x128

m = M().eval()
x126 = torch.randn(torch.Size([1, 128, 14, 14]))
x111 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x126, x111)
end = time.time()
print(end-start)
