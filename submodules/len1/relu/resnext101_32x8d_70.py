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
        self.relu70 = ReLU(inplace=True)

    def forward(self, x242):
        x243=self.relu70(x242)
        return x243

m = M().eval()
x242 = torch.randn(torch.Size([1, 1024, 14, 14]))
start = time.time()
output = m(x242)
end = time.time()
print(end-start)
