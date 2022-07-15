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
        self.relu40 = ReLU(inplace=True)

    def forward(self, x168):
        x169=self.relu40(x168)
        return x169

m = M().eval()
x168 = torch.randn(torch.Size([1, 1232, 14, 14]))
start = time.time()
output = m(x168)
end = time.time()
print(end-start)
