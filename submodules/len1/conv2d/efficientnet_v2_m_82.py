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
        self.conv2d82 = Conv2d(44, 1056, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x265):
        x266=self.conv2d82(x265)
        return x266

m = M().eval()
x265 = torch.randn(torch.Size([1, 44, 1, 1]))
start = time.time()
output = m(x265)
end = time.time()
print(end-start)
