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
        self.relu91 = ReLU(inplace=True)

    def forward(self, x315):
        x316=self.relu91(x315)
        return x316

m = M().eval()
x315 = torch.randn(torch.Size([1, 256, 14, 14]))
start = time.time()
output = m(x315)
end = time.time()
print(end-start)