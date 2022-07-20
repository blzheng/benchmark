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
        self.relu623 = ReLU6(inplace=True)

    def forward(self, x99):
        x100=self.relu623(x99)
        return x100

m = M().eval()
x99 = torch.randn(torch.Size([1, 576, 14, 14]))
start = time.time()
output = m(x99)
end = time.time()
print(end-start)