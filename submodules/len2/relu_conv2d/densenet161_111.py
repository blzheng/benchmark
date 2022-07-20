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
        self.relu112 = ReLU(inplace=True)
        self.conv2d112 = Conv2d(1056, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x399):
        x400=self.relu112(x399)
        x401=self.conv2d112(x400)
        return x401

m = M().eval()
x399 = torch.randn(torch.Size([1, 1056, 7, 7]))
start = time.time()
output = m(x399)
end = time.time()
print(end-start)
