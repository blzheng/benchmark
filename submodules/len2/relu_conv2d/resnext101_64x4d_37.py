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
        self.relu37 = ReLU(inplace=True)
        self.conv2d42 = Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x135):
        x136=self.relu37(x135)
        x137=self.conv2d42(x136)
        return x137

m = M().eval()
x135 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x135)
end = time.time()
print(end-start)
