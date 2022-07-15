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
        self.relu132 = ReLU(inplace=True)

    def forward(self, x467):
        x468=self.relu132(x467)
        return x468

m = M().eval()
x467 = torch.randn(torch.Size([1, 128, 14, 14]))
start = time.time()
output = m(x467)
end = time.time()
print(end-start)
