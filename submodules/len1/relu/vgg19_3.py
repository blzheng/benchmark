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
        self.relu3 = ReLU(inplace=True)

    def forward(self, x8):
        x9=self.relu3(x8)
        return x9

m = M().eval()
x8 = torch.randn(torch.Size([1, 128, 112, 112]))
start = time.time()
output = m(x8)
end = time.time()
print(end-start)
