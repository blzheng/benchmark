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
        self.relu6 = ReLU(inplace=True)

    def forward(self, x24):
        x25=self.relu6(x24)
        return x25

m = M().eval()
x24 = torch.randn(torch.Size([1, 72, 56, 56]))
start = time.time()
output = m(x24)
end = time.time()
print(end-start)
