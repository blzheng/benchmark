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
        self.relu97 = ReLU(inplace=True)

    def forward(self, x340, x332):
        x341=operator.add(x340, x332)
        x342=self.relu97(x341)
        return x342

m = M().eval()
x340 = torch.randn(torch.Size([1, 2048, 7, 7]))
x332 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x340, x332)
end = time.time()
print(end-start)
