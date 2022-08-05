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
        self.relu2 = ReLU(inplace=True)

    def forward(self, x14):
        x15=self.relu2(x14)
        return x15

m = M().eval()
x14 = torch.randn(torch.Size([1, 64, 56, 56]))
start = time.time()
output = m(x14)
end = time.time()
print(end-start)
