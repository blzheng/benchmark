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
        self.relu29 = ReLU(inplace=True)

    def forward(self, x113, x108):
        x114=operator.add(x113, x108)
        x115=self.relu29(x114)
        return x115

m = M().eval()
x113 = torch.randn(torch.Size([1, 512, 7, 7]))
x108 = torch.randn(torch.Size([1, 512, 7, 7]))
start = time.time()
output = m(x113, x108)
end = time.time()
print(end-start)
