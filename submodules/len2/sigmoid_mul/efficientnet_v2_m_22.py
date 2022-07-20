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

    def forward(self, x440, x436):
        x441=self.sigmoid22(x440)
        x442=operator.mul(x441, x436)
        return x442

m = M().eval()
x440 = torch.randn(torch.Size([1, 1824, 1, 1]))
x436 = torch.randn(torch.Size([1, 1824, 7, 7]))
start = time.time()
output = m(x440, x436)
end = time.time()
print(end-start)
