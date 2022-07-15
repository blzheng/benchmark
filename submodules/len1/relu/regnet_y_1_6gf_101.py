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
        self.relu101 = ReLU(inplace=True)

    def forward(self, x413):
        x414=self.relu101(x413)
        return x414

m = M().eval()
x413 = torch.randn(torch.Size([1, 888, 14, 14]))
start = time.time()
output = m(x413)
end = time.time()
print(end-start)
