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
        self.conv2d2 = Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x5):
        x6=self.conv2d2(x5)
        return x6

m = M().eval()
x5 = torch.randn(torch.Size([1, 16, 54, 54]))
start = time.time()
output = m(x5)
end = time.time()
print(end-start)
