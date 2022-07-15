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
        self.sigmoid21 = Sigmoid()

    def forward(self, x355):
        x356=self.sigmoid21(x355)
        return x356

m = M().eval()
x355 = torch.randn(torch.Size([1, 2904, 1, 1]))
start = time.time()
output = m(x355)
end = time.time()
print(end-start)
