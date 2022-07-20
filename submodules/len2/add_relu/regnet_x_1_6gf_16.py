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
        self.relu51 = ReLU(inplace=True)

    def forward(self, x171, x179):
        x180=operator.add(x171, x179)
        x181=self.relu51(x180)
        return x181

m = M().eval()
x171 = torch.randn(torch.Size([1, 912, 7, 7]))
x179 = torch.randn(torch.Size([1, 912, 7, 7]))
start = time.time()
output = m(x171, x179)
end = time.time()
print(end-start)
