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
        self.conv2d57 = Conv2d(672, 28, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x167):
        x168=x167.mean((2, 3),keepdim=True)
        x169=self.conv2d57(x168)
        return x169

m = M().eval()
x167 = torch.randn(torch.Size([1, 672, 7, 7]))
start = time.time()
output = m(x167)
end = time.time()
print(end-start)