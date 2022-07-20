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
        self.relu47 = ReLU(inplace=True)
        self.conv2d52 = Conv2d(672, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x166):
        x167=self.relu47(x166)
        x168=self.conv2d52(x167)
        return x168

m = M().eval()
x166 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
