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
        self.sigmoid38 = Sigmoid()

    def forward(self, x600, x596):
        x601=self.sigmoid38(x600)
        x602=operator.mul(x601, x596)
        return x602

m = M().eval()
x600 = torch.randn(torch.Size([1, 2064, 1, 1]))
x596 = torch.randn(torch.Size([1, 2064, 7, 7]))
start = time.time()
output = m(x600, x596)
end = time.time()
print(end-start)
