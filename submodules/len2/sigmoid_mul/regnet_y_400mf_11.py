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
        self.sigmoid11 = Sigmoid()

    def forward(self, x197, x193):
        x198=self.sigmoid11(x197)
        x199=operator.mul(x198, x193)
        return x199

m = M().eval()
x197 = torch.randn(torch.Size([1, 440, 1, 1]))
x193 = torch.randn(torch.Size([1, 440, 7, 7]))
start = time.time()
output = m(x197, x193)
end = time.time()
print(end-start)
