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
        self.sigmoid51 = Sigmoid()

    def forward(self, x937):
        x938=self.sigmoid51(x937)
        return x938

m = M().eval()
x937 = torch.randn(torch.Size([1, 2304, 1, 1]))
start = time.time()
output = m(x937)
end = time.time()
print(end-start)
