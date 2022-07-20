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

    def forward(self, x391, x387):
        x392=x391.sigmoid()
        x393=operator.mul(x387, x392)
        return x393

m = M().eval()
x391 = torch.randn(torch.Size([1, 960, 1, 1]))
x387 = torch.randn(torch.Size([1, 960, 14, 14]))
start = time.time()
output = m(x391, x387)
end = time.time()
print(end-start)
