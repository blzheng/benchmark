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
        self.relu21 = ReLU(inplace=True)

    def forward(self, x84):
        x85=self.relu21(x84)
        return x85

m = M().eval()
x84 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x84)
end = time.time()
print(end-start)
