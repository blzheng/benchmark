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
        self.sigmoid12 = Sigmoid()

    def forward(self, x263, x259):
        x264=self.sigmoid12(x263)
        x265=operator.mul(x264, x259)
        return x265

m = M().eval()
x263 = torch.randn(torch.Size([1, 960, 1, 1]))
x259 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x263, x259)
end = time.time()
print(end-start)
