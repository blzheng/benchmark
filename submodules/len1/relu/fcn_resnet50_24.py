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
        self.relu22 = ReLU(inplace=True)

    def forward(self, x91):
        x92=self.relu22(x91)
        return x92

m = M().eval()
x91 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x91)
end = time.time()
print(end-start)
