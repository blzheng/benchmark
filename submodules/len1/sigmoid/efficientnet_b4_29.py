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
        self.sigmoid29 = Sigmoid()

    def forward(self, x459):
        x460=self.sigmoid29(x459)
        return x460

m = M().eval()
x459 = torch.randn(torch.Size([1, 1632, 1, 1]))
start = time.time()
output = m(x459)
end = time.time()
print(end-start)
