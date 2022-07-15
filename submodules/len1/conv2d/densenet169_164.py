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
        self.conv2d164 = Conv2d(1600, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x582):
        x583=self.conv2d164(x582)
        return x583

m = M().eval()
x582 = torch.randn(torch.Size([1, 1600, 7, 7]))
start = time.time()
output = m(x582)
end = time.time()
print(end-start)
