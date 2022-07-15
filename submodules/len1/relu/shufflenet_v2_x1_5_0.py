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
        self.relu0 = ReLU(inplace=True)

    def forward(self, x2):
        x3=self.relu0(x2)
        return x3

m = M().eval()
x2 = torch.randn(torch.Size([1, 24, 112, 112]))
start = time.time()
output = m(x2)
end = time.time()
print(end-start)
