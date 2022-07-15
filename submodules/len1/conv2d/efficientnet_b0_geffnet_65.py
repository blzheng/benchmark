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
        self.conv2d65 = Conv2d(192, 1152, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x190):
        x191=self.conv2d65(x190)
        return x191

m = M().eval()
x190 = torch.randn(torch.Size([1, 192, 7, 7]))
start = time.time()
output = m(x190)
end = time.time()
print(end-start)
