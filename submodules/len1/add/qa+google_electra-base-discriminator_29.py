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

    def forward(self, x194, x190):
        x195=operator.add(x194, x190)
        return x195

m = M().eval()
x194 = torch.randn(torch.Size([1, 384, 768]))
x190 = torch.randn(torch.Size([1, 384, 768]))
start = time.time()
output = m(x194, x190)
end = time.time()
print(end-start)
