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
        self.sigmoid0 = Sigmoid()

    def forward(self, x127, x123):
        x128=self.sigmoid0(x127)
        x129=operator.mul(x128, x123)
        return x129

m = M().eval()
x127 = torch.randn(torch.Size([1, 384, 1, 1]))
x123 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x127, x123)
end = time.time()
print(end-start)
