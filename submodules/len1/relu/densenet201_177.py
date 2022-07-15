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
        self.relu177 = ReLU(inplace=True)

    def forward(self, x626):
        x627=self.relu177(x626)
        return x627

m = M().eval()
x626 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x626)
end = time.time()
print(end-start)
