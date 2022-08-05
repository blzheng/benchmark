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

    def forward(self, x180, x194):
        x195=operator.add(x180, x194)
        return x195

m = M().eval()
x180 = torch.randn(torch.Size([1, 14, 14, 512]))
x194 = torch.randn(torch.Size([1, 14, 14, 512]))
start = time.time()
output = m(x180, x194)
end = time.time()
print(end-start)
