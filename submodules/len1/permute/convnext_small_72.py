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

    def forward(self, x381):
        x382=x381.permute(0, 2, 3, 1)
        return x382

m = M().eval()
x381 = torch.randn(torch.Size([1, 384, 14, 14]))
start = time.time()
output = m(x381)
end = time.time()
print(end-start)
