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

    def forward(self, x127, x123):
        x128=x127.sigmoid()
        x129=operator.mul(x123, x128)
        return x129

m = M().eval()
x127 = torch.randn(torch.Size([1, 480, 1, 1]))
x123 = torch.randn(torch.Size([1, 480, 14, 14]))
start = time.time()
output = m(x127, x123)
end = time.time()
print(end-start)
