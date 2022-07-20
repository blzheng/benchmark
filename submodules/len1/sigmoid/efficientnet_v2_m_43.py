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
        self.sigmoid43 = Sigmoid()

    def forward(self, x774):
        x775=self.sigmoid43(x774)
        return x775

m = M().eval()
x774 = torch.randn(torch.Size([1, 3072, 1, 1]))
start = time.time()
output = m(x774)
end = time.time()
print(end-start)