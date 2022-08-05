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
        self.relu40 = ReLU(inplace=True)
        self.conv2d46 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x151):
        x152=self.relu40(x151)
        x153=self.conv2d46(x152)
        return x153

m = M().eval()
x151 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x151)
end = time.time()
print(end-start)
