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
        self.relu158 = ReLU(inplace=True)

    def forward(self, x560):
        x561=self.relu158(x560)
        return x561

m = M().eval()
x560 = torch.randn(torch.Size([1, 2160, 7, 7]))
start = time.time()
output = m(x560)
end = time.time()
print(end-start)
