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
        self.conv2d28 = Conv2d(320, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x95, x90):
        x96=operator.mul(x95, x90)
        x97=self.conv2d28(x96)
        return x97

m = M().eval()
x95 = torch.randn(torch.Size([1, 320, 1, 1]))
x90 = torch.randn(torch.Size([1, 320, 14, 14]))
start = time.time()
output = m(x95, x90)
end = time.time()
print(end-start)
