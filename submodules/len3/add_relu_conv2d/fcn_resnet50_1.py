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
        self.relu4 = ReLU(inplace=True)
        self.conv2d8 = Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x26, x18):
        x27=operator.add(x26, x18)
        x28=self.relu4(x27)
        x29=self.conv2d8(x28)
        return x29

m = M().eval()
x26 = torch.randn(torch.Size([1, 256, 56, 56]))
x18 = torch.randn(torch.Size([1, 256, 56, 56]))
start = time.time()
output = m(x26, x18)
end = time.time()
print(end-start)
