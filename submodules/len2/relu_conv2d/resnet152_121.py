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
        self.relu121 = ReLU(inplace=True)
        self.conv2d126 = Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x415):
        x416=self.relu121(x415)
        x417=self.conv2d126(x416)
        return x417

m = M().eval()
x415 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x415)
end = time.time()
print(end-start)
