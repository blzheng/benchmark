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
        self.relu57 = ReLU(inplace=True)

    def forward(self, x191, x199):
        x200=operator.add(x191, x199)
        x201=self.relu57(x200)
        return x201

m = M().eval()
x191 = torch.randn(torch.Size([1, 400, 7, 7]))
x199 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x191, x199)
end = time.time()
print(end-start)
