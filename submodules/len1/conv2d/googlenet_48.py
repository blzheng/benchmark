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
        self.conv2d48 = Conv2d(832, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x166):
        x176=self.conv2d48(x166)
        return x176

m = M().eval()
x166 = torch.randn(torch.Size([1, 832, 7, 7]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
