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
        self.relu141 = ReLU(inplace=True)

    def forward(self, x500):
        x501=self.relu141(x500)
        return x501

m = M().eval()
x500 = torch.randn(torch.Size([1, 128, 7, 7]))
start = time.time()
output = m(x500)
end = time.time()
print(end-start)
