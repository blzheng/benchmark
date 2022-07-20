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
        self.relu4 = ReLU(inplace=True)

    def forward(self, x5, x19):
        x20=operator.add(x5, x19)
        x21=self.relu4(x20)
        return x21

m = M().eval()
x5 = torch.randn(torch.Size([1, 232, 56, 56]))
x19 = torch.randn(torch.Size([1, 232, 56, 56]))
start = time.time()
output = m(x5, x19)
end = time.time()
print(end-start)
