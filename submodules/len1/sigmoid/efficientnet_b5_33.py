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
        self.sigmoid33 = Sigmoid()

    def forward(self, x520):
        x521=self.sigmoid33(x520)
        return x521

m = M().eval()
x520 = torch.randn(torch.Size([1, 1824, 1, 1]))
start = time.time()
output = m(x520)
end = time.time()
print(end-start)
