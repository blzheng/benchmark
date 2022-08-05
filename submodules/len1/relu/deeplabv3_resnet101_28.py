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
        self.relu28 = ReLU(inplace=True)

    def forward(self, x104):
        x105=self.relu28(x104)
        return x105

m = M().eval()
x104 = torch.randn(torch.Size([1, 256, 28, 28]))
start = time.time()
output = m(x104)
end = time.time()
print(end-start)
