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
        self.sigmoid44 = Sigmoid()

    def forward(self, x694):
        x695=self.sigmoid44(x694)
        return x695

m = M().eval()
x694 = torch.randn(torch.Size([1, 3456, 1, 1]))
start = time.time()
output = m(x694)
end = time.time()
print(end-start)
