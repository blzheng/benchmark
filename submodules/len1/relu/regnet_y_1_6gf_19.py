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
        self.relu19 = ReLU()

    def forward(self, x79):
        x80=self.relu19(x79)
        return x80

m = M().eval()
x79 = torch.randn(torch.Size([1, 30, 1, 1]))
start = time.time()
output = m(x79)
end = time.time()
print(end-start)