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
        self.conv2d23 = Conv2d(244, 244, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x143):
        x144=self.conv2d23(x143)
        return x144

m = M().eval()
x143 = torch.randn(torch.Size([1, 244, 14, 14]))
start = time.time()
output = m(x143)
end = time.time()
print(end-start)
