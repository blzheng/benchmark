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
        self.sigmoid22 = Sigmoid()

    def forward(self, x345, x341):
        x346=self.sigmoid22(x345)
        x347=operator.mul(x346, x341)
        return x347

m = M().eval()
x345 = torch.randn(torch.Size([1, 960, 1, 1]))
x341 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x345, x341)
end = time.time()
print(end-start)
