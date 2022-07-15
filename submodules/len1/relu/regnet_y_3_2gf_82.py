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
        self.relu82 = ReLU(inplace=True)

    def forward(self, x336):
        x337=self.relu82(x336)
        return x337

m = M().eval()
x336 = torch.randn(torch.Size([1, 1512, 7, 7]))
start = time.time()
output = m(x336)
end = time.time()
print(end-start)
