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
        self.conv2d41 = Conv2d(320, 80, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x128):
        x129=self.conv2d41(x128)
        return x129

m = M().eval()
x128 = torch.randn(torch.Size([1, 320, 1, 1]))
start = time.time()
output = m(x128)
end = time.time()
print(end-start)
