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
        self.sigmoid55 = Sigmoid()

    def forward(self, x999):
        x1000=self.sigmoid55(x999)
        return x1000

m = M().eval()
x999 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x999)
end = time.time()
print(end-start)
