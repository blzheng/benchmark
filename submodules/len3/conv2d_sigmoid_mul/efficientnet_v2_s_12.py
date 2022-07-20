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
        self.conv2d82 = Conv2d(40, 960, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid12 = Sigmoid()

    def forward(self, x262, x259):
        x263=self.conv2d82(x262)
        x264=self.sigmoid12(x263)
        x265=operator.mul(x264, x259)
        return x265

m = M().eval()
x262 = torch.randn(torch.Size([1, 40, 1, 1]))
x259 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x262, x259)
end = time.time()
print(end-start)
