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
        self.relu148 = ReLU(inplace=True)

    def forward(self, x510, x502):
        x511=operator.add(x510, x502)
        x512=self.relu148(x511)
        return x512

m = M().eval()
x510 = torch.randn(torch.Size([1, 2048, 7, 7]))
x502 = torch.randn(torch.Size([1, 2048, 7, 7]))
start = time.time()
output = m(x510, x502)
end = time.time()
print(end-start)
