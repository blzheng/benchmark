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
        self.conv2d56 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x166):
        x167=x166.mean((2, 3),keepdim=True)
        x168=self.conv2d56(x167)
        return x168

m = M().eval()
x166 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x166)
end = time.time()
print(end-start)
