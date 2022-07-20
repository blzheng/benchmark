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
        self.relu36 = ReLU(inplace=True)
        self.conv2d40 = Conv2d(1344, 1344, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x128):
        x129=self.relu36(x128)
        x130=self.conv2d40(x129)
        return x130

m = M().eval()
x128 = torch.randn(torch.Size([1, 1344, 14, 14]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
