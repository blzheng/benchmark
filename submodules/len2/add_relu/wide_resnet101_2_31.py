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
        self.relu94 = ReLU(inplace=True)

    def forward(self, x330, x322):
        x331=operator.add(x330, x322)
        x332=self.relu94(x331)
        return x332

m = M().eval()
x330 = torch.randn(torch.Size([1, 2048, 7, 7]))
x322 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x330, x322)
end = time.time()
print(end-start)
