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
        self.conv2d266 = Conv2d(3840, 640, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x837):
        x838=self.conv2d266(x837)
        return x838

m = M().eval()
x837 = torch.randn(torch.Size([1, 3840, 7, 7]))
start = time.time()
output = m(x837)
end = time.time()
print(end-start)
