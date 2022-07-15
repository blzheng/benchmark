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
        self.relu75 = ReLU()

    def forward(self, x305):
        x306=self.relu75(x305)
        return x306

m = M().eval()
x305 = torch.randn(torch.Size([1, 84, 1, 1]))
start = time.time()
output = m(x305)
end = time.time()
print(end-start)
