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
        self.relu46 = ReLU(inplace=True)

    def forward(self, x170, x162):
        x171=operator.add(x170, x162)
        x172=self.relu46(x171)
        return x172

m = M().eval()
x170 = torch.randn(torch.Size([1, 1024, 28, 28]))
x162 = torch.randn(torch.Size([1, 1024, 28, 28]))
start = time.time()
output = m(x170, x162)
end = time.time()
print(end-start)
