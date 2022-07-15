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
        self.sigmoid59 = Sigmoid()

    def forward(self, x1063):
        x1064=self.sigmoid59(x1063)
        return x1064

m = M().eval()
x1063 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x1063)
end = time.time()
print(end-start)
