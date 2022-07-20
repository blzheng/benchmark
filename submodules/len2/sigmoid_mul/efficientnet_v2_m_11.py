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
        self.sigmoid11 = Sigmoid()

    def forward(self, x266, x262):
        x267=self.sigmoid11(x266)
        x268=operator.mul(x267, x262)
        return x268

m = M().eval()
x266 = torch.randn(torch.Size([1, 1056, 1, 1]))
x262 = torch.randn(torch.Size([1, 1056, 14, 14]))
start = time.time()
output = m(x266, x262)
end = time.time()
print(end-start)
