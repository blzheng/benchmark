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
        self.relu28 = ReLU(inplace=True)
        self.conv2d28 = Conv2d(528, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x102):
        x103=self.relu28(x102)
        x104=self.conv2d28(x103)
        return x104

m = M().eval()
x102 = torch.randn(torch.Size([1, 528, 28, 28]))
start = time.time()
output = m(x102)
end = time.time()
print(end-start)
