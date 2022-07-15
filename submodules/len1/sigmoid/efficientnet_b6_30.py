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
        self.sigmoid30 = Sigmoid()

    def forward(self, x474):
        x475=self.sigmoid30(x474)
        return x475

m = M().eval()
x474 = torch.randn(torch.Size([1, 1200, 1, 1]))
start = time.time()
output = m(x474)
end = time.time()
print(end-start)
