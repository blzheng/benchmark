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
        self.relu10 = ReLU()

    def forward(self, x50):
        x51=self.relu10(x50)
        return x51

m = M().eval()
x50 = torch.randn(torch.Size([1, 32, 1, 1]))
start = time.time()
output = m(x50)
end = time.time()
print(end-start)
