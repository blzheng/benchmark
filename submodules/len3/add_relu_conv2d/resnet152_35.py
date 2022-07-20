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
        self.relu106 = ReLU(inplace=True)
        self.conv2d112 = Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x368, x360):
        x369=operator.add(x368, x360)
        x370=self.relu106(x369)
        x371=self.conv2d112(x370)
        return x371

m = M().eval()
x368 = torch.randn(torch.Size([1, 1024, 14, 14]))
x360 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x368, x360)
end = time.time()
print(end-start)
