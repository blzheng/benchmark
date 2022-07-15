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
        self.conv2d78 = Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)

    def forward(self, x241):
        x242=self.conv2d78(x241)
        return x242

m = M().eval()
x241 = torch.randn(torch.Size([1, 672, 14, 14]))
start = time.time()
output = m(x241)
end = time.time()
print(end-start)
