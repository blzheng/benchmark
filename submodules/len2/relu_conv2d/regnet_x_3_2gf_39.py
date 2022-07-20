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
        self.relu39 = ReLU(inplace=True)
        self.conv2d43 = Conv2d(432, 432, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x138):
        x139=self.relu39(x138)
        x140=self.conv2d43(x139)
        return x140

m = M().eval()
x138 = torch.randn(torch.Size([1, 432, 14, 14]))
start = time.time()
output = m(x138)
end = time.time()
print(end-start)
