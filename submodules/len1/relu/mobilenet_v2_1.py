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
        self.relu61 = ReLU6(inplace=True)

    def forward(self, x5):
        x6=self.relu61(x5)
        return x6

m = M().eval()
x5 = torch.randn(torch.Size([1, 32, 112, 112]))
start = time.time()
output = m(x5)
end = time.time()
print(end-start)