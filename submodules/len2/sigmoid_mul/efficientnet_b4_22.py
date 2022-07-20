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

    def forward(self, x349, x345):
        x350=self.sigmoid22(x349)
        x351=operator.mul(x350, x345)
        return x351

m = M().eval()
x349 = torch.randn(torch.Size([1, 960, 1, 1]))
x345 = torch.randn(torch.Size([1, 960, 7, 7]))
start = time.time()
output = m(x349, x345)
end = time.time()
print(end-start)
