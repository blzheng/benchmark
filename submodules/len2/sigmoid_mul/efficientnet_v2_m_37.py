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
        self.sigmoid37 = Sigmoid()

    def forward(self, x680, x676):
        x681=self.sigmoid37(x680)
        x682=operator.mul(x681, x676)
        return x682

m = M().eval()
x680 = torch.randn(torch.Size([1, 1824, 1, 1]))
x676 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x680, x676)
end = time.time()
print(end-start)
