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

    def forward(self, x660, x656):
        x661=x660.sigmoid()
        x662=operator.mul(x656, x661)
        return x662

m = M().eval()
x660 = torch.randn(torch.Size([1, 3456, 1, 1]))
x656 = torch.randn(torch.Size([1, 3456, 7, 7]))
start = time.time()
output = m(x660, x656)
end = time.time()
print(end-start)
