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
        self.relu29 = ReLU(inplace=True)
        self.conv2d44 = Conv2d(1152, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x125):
        x126=self.relu29(x125)
        x127=self.conv2d44(x126)
        return x127

m = M().eval()
x125 = torch.randn(torch.Size([1, 1152, 7, 7]))
start = time.time()
output = m(x125)
end = time.time()
print(end-start)
