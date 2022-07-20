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
        self.relu13 = ReLU(inplace=True)

    def forward(self, x55, x57):
        x58=operator.add(x55, x57)
        x59=self.relu13(x58)
        return x59

m = M().eval()
x55 = torch.randn(torch.Size([1, 512, 7, 7]))
x57 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x55, x57)
end = time.time()
print(end-start)
