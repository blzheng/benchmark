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
        self.relu60 = ReLU(inplace=True)

    def forward(self, x201, x209):
        x210=operator.add(x201, x209)
        x211=self.relu60(x210)
        return x211

m = M().eval()
x201 = torch.randn(torch.Size([1, 400, 7, 7]))
x209 = torch.randn(torch.Size([1, 400, 7, 7]))
start = time.time()
output = m(x201, x209)
end = time.time()
print(end-start)
