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
        self.sigmoid58 = Sigmoid()

    def forward(self, x1047):
        x1048=self.sigmoid58(x1047)
        return x1048

m = M().eval()
x1047 = torch.randn(torch.Size([1, 3840, 1, 1]))
start = time.time()
output = m(x1047)
end = time.time()
print(end-start)
