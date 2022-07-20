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
        self.relu94 = ReLU(inplace=True)

    def forward(self, x382):
        x383=self.relu94(x382)
        return x383

m = M().eval()
x382 = torch.randn(torch.Size([1, 2904, 14, 14]))
start = time.time()
output = m(x382)
end = time.time()
print(end-start)