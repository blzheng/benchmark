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

    def forward(self, x809):
        x810=self.sigmoid43(x809)
        return x810

m = M().eval()
x809 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x809)
end = time.time()
print(end-start)
